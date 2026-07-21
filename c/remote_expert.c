#define _GNU_SOURCE
#include "remote_expert.h"
#include "remote_expert_protocol.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    union ibv_gid gid;
    uint32_t qpn, psn;
    uint64_t target_addr;
    uint32_t target_rkey;
} PeerInfo;

typedef struct {
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *request_mr, *response_mr;
    void *request, *response;
    PeerInfo self, peer;
    int tcp, gid_index, connected;
} Remote;

static Remote g_remote;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned char g_layers[128];
static char g_host[256], g_device[64];
static int g_port = 19066, g_timeout_ms = 5000, g_configured;
static uint32_t g_seq;
static uint64_t g_calls, g_experts, g_rows, g_bytes, g_us, g_fallbacks;

static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

static int send_all(int fd, const void *data, size_t size) {
    const char *p = data;
    while (size) {
        ssize_t n = send(fd, p, size, MSG_NOSIGNAL);
        if (n <= 0) return 0;
        p += n;
        size -= (size_t)n;
    }
    return 1;
}

static int recv_all(int fd, void *data, size_t size) {
    char *p = data;
    while (size) {
        ssize_t n = recv(fd, p, size, MSG_WAITALL);
        if (n <= 0) return 0;
        p += n;
        size -= (size_t)n;
    }
    return 1;
}

static void cleanup(void) {
    Remote *r = &g_remote;
    if (r->request_mr) ibv_dereg_mr(r->request_mr);
    if (r->response_mr) ibv_dereg_mr(r->response_mr);
    if (r->qp) ibv_destroy_qp(r->qp);
    if (r->cq) ibv_destroy_cq(r->cq);
    if (r->pd) ibv_dealloc_pd(r->pd);
    if (r->context) ibv_close_device(r->context);
    if (r->tcp >= 0) close(r->tcp);
    free(r->request);
    free(r->response);
    memset(r, 0, sizeof(*r));
    r->tcp = -1;
}

static void disable_remote(const char *message) {
    fprintf(stderr, "[REMOTE] disabled: %s\n", message);
    cleanup();
    g_configured = 0;
    g_fallbacks++;
}

static int connect_tcp(void) {
    struct addrinfo hints = {0}, *addresses = NULL;
    char port[16];
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(port, sizeof(port), "%d", g_port);
    int gai = getaddrinfo(g_host, port, &hints, &addresses);
    if (gai) {
        fprintf(stderr, "[REMOTE] getaddrinfo %s: %s\n", g_host, gai_strerror(gai));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *a = addresses; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = connect(fd, a->ai_addr, a->ai_addrlen);
        if (rc && errno == EINPROGRESS) {
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            rc = poll(&pfd, 1, g_timeout_ms);
            if (rc > 0) {
                int error = 0;
                socklen_t length = sizeof(error);
                rc = getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length);
                if (!rc && error) { errno = error; rc = -1; }
            } else rc = -1;
        }
        if (!rc) {
            if (flags >= 0) fcntl(fd, F_SETFL, flags);
            struct timeval timeout = {
                .tv_sec = g_timeout_ms / 1000,
                .tv_usec = (g_timeout_ms % 1000) * 1000,
            };
            if (!setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) &&
                !setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)))
                break;
            rc = -1;
        }
        if (fd >= 0) close(fd);
        fd = -1;
    }
    freeaddrinfo(addresses);
    return fd;
}

static int modify_qp(Remote *r) {
    struct ibv_qp_attr attr = {0};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(r->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                    IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
        return 0;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = r->peer.qpn;
    attr.rq_psn = r->peer.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.grh.dgid = r->peer.gid;
    attr.ah_attr.grh.sgid_index = r->gid_index;
    attr.ah_attr.grh.hop_limit = 64;
    if (ibv_modify_qp(r->qp, &attr, IBV_QP_STATE | IBV_QP_AV |
                                    IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                                    IBV_QP_MIN_RNR_TIMER))
        return 0;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = r->self.psn;
    attr.max_rd_atomic = 1;
    return !ibv_modify_qp(r->qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT |
                                         IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                         IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

static int remote_connect(void) {
    Remote *r = &g_remote;
    int ndev = 0;
    struct ibv_device **devices = ibv_get_device_list(&ndev);
    struct ibv_device *device = NULL;
    for (int i = 0; i < ndev; i++)
        if (!strcmp(ibv_get_device_name(devices[i]), g_device)) device = devices[i];
    if (!device) {
        ibv_free_device_list(devices);
        disable_remote("RDMA device not found");
        return 0;
    }
    r->context = ibv_open_device(device);
    ibv_free_device_list(devices);
    r->pd = r->context ? ibv_alloc_pd(r->context) : NULL;
    r->cq = r->context ? ibv_create_cq(r->context, 16, NULL, NULL, 0) : NULL;
    struct ibv_qp_init_attr init = {0};
    init.send_cq = init.recv_cq = r->cq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = init.cap.max_recv_wr = 16;
    init.cap.max_send_sge = init.cap.max_recv_sge = 1;
    r->qp = r->pd ? ibv_create_qp(r->pd, &init) : NULL;
    if (!r->context || !r->pd || !r->cq || !r->qp ||
        posix_memalign(&r->request, 4096, COLI_REMOTE_REQUEST_BYTES) ||
        posix_memalign(&r->response, 4096, COLI_REMOTE_RESPONSE_BYTES)) {
        disable_remote("RDMA allocation failed");
        return 0;
    }
    r->request_mr = ibv_reg_mr(r->pd, r->request, COLI_REMOTE_REQUEST_BYTES, 0);
    r->response_mr = ibv_reg_mr(r->pd, r->response, COLI_REMOTE_RESPONSE_BYTES,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!r->request_mr || !r->response_mr ||
        ibv_query_gid(r->context, 1, r->gid_index, &r->self.gid)) {
        disable_remote("RDMA registration failed");
        return 0;
    }
    r->self.qpn = r->qp->qp_num;
    r->self.psn = (uint32_t)(getpid() * 2654435761u) & 0xffffffu;
    r->self.target_addr = (uintptr_t)r->response;
    r->self.target_rkey = r->response_mr->rkey;
    r->tcp = connect_tcp();
    if (r->tcp < 0 || !send_all(r->tcp, &r->self, sizeof(r->self)) ||
        !recv_all(r->tcp, &r->peer, sizeof(r->peer)) || !modify_qp(r)) {
        disable_remote("connection failed");
        return 0;
    }
    char ready = 1, peer_ready = 0;
    if (!send_all(r->tcp, &ready, 1) || !recv_all(r->tcp, &peer_ready, 1)) {
        disable_remote("connection handshake failed");
        return 0;
    }
    r->connected = 1;
    fprintf(stderr, "[REMOTE] connected to %s via %s GID %d\n",
            g_host, g_device, r->gid_index);
    return 1;
}

static int post_receive(Remote *r, uint64_t id) {
    struct ibv_recv_wr wr = {0}, *bad = NULL;
    wr.wr_id = id;
    return !ibv_post_recv(r->qp, &wr, &bad);
}

static int write_request(Remote *r, size_t size, uint64_t id) {
    struct ibv_sge sge = {
        .addr = (uintptr_t)r->request,
        .length = (uint32_t)size,
        .lkey = r->request_mr->lkey,
    };
    struct ibv_send_wr wr = {0}, *bad = NULL;
    wr.wr_id = id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = htonl((uint32_t)id);
    wr.wr.rdma.remote_addr = r->peer.target_addr;
    wr.wr.rdma.rkey = r->peer.target_rkey;
    return !ibv_post_send(r->qp, &wr, &bad);
}

static int wait_for(Remote *r, uint64_t id) {
    uint64_t deadline = now_us() + (uint64_t)g_timeout_ms * 1000u;
    for (;;) {
        struct ibv_wc completion;
        int n = ibv_poll_cq(r->cq, 1, &completion);
        if (n < 0) return 0;
        if (!n) {
            if (now_us() >= deadline) return 0;
            continue;
        }
        if (completion.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "[REMOTE] completion: %s\n",
                    ibv_wc_status_str(completion.status));
            return 0;
        }
        if (completion.wr_id == id) return 1;
    }
}

static void configure(void) {
    g_remote.tcp = -1;
    const char *host = getenv("COLI_REMOTE_EXPERT");
    const char *layers = getenv("COLI_REMOTE_LAYERS");
    if (!host || !*host || !layers || !*layers) return;
    snprintf(g_host, sizeof(g_host), "%s", host);
    const char *device = getenv("COLI_REMOTE_DEVICE");
    snprintf(g_device, sizeof(g_device), "%s", device ? device : "rocep1s0f1");
    const char *port = getenv("COLI_REMOTE_PORT");
    if (port) g_port = atoi(port);
    const char *timeout = getenv("COLI_REMOTE_TIMEOUT_MS");
    if (timeout) g_timeout_ms = atoi(timeout);
    if (g_timeout_ms < 100) g_timeout_ms = 100;
    g_remote.gid_index = getenv("COLI_REMOTE_GID") ?
                         atoi(getenv("COLI_REMOTE_GID")) : 3;
    const char *p = layers;
    while (*p) {
        char *end = NULL;
        long layer = strtol(p, &end, 10);
        if (end == p || layer < 0 || layer >= (long)sizeof(g_layers)) {
            fprintf(stderr, "[REMOTE] invalid COLI_REMOTE_LAYERS=%s\n", layers);
            return;
        }
        g_layers[layer] = 1;
        p = end;
        if (*p == ',') p++;
        else if (*p) {
            fprintf(stderr, "[REMOTE] invalid COLI_REMOTE_LAYERS=%s\n", layers);
            memset(g_layers, 0, sizeof(g_layers));
            return;
        }
    }
    g_configured = 1;
    atexit(coli_remote_expert_shutdown);
}

int coli_remote_expert_enabled(int layer, int S) {
    pthread_once(&g_once, configure);
    return g_configured && S > 0 && S <= COLI_REMOTE_MAX_S &&
           layer >= 0 && layer < (int)sizeof(g_layers) && g_layers[layer];
}

int coli_remote_expert_run(int layer, const float *x, int S, const int *uniq,
                           int nu, const int *idxs, const float *weights,
                           const int *keff, int K, int D, float *out) {
    if (!coli_remote_expert_enabled(layer, S) || D != COLI_REMOTE_HIDDEN ||
        nu < 1 || nu > COLI_REMOTE_MAX_EXPERTS)
        return 0;
    pthread_mutex_lock(&g_lock);
    Remote *r = &g_remote;
    if (!r->connected && !remote_connect()) {
        pthread_mutex_unlock(&g_lock);
        return 0;
    }
    ColiRemoteRequest *request = r->request;
    memset(request, 0, sizeof(*request));
    request->magic = COLI_REMOTE_MAGIC;
    request->version = COLI_REMOTE_VERSION;
    request->seq = ++g_seq;
    request->layer = (uint32_t)layer;
    request->S = (uint32_t)S;
    request->count = (uint32_t)nu;
    int total = 0;
    for (int j = 0; j < nu; j++) {
        request->eids[j] = uniq[j];
        for (int s = 0; s < S; s++)
            for (int k = 0; k < keff[s]; k++)
                if (idxs[(size_t)s * K + k] == uniq[j]) {
                    if (total >= COLI_REMOTE_MAX_ROWS) {
                        disable_remote("route count exceeds protocol");
                        pthread_mutex_unlock(&g_lock);
                        return 0;
                    }
                    request->tokrow[total++] = s;
                    request->rows[j]++;
                    break;
                }
    }
    request->total_rows = (uint32_t)total;
    float *request_x = (float *)(request + 1);
    memcpy(request_x, x, (size_t)S * D * sizeof(float));
    size_t request_bytes = sizeof(*request) + (size_t)S * D * sizeof(float);
    uint64_t started = now_us();
    if (!post_receive(r, 2) || !write_request(r, request_bytes, 1) ||
        !wait_for(r, 2)) {
        disable_remote("request failed");
        pthread_mutex_unlock(&g_lock);
        return 0;
    }
    ColiRemoteResponse *response = r->response;
    if (response->magic != COLI_REMOTE_MAGIC ||
        response->version != COLI_REMOTE_VERSION ||
        response->seq != request->seq || response->status ||
        response->count != request->count ||
        response->total_rows != request->total_rows) {
        disable_remote("invalid worker response");
        pthread_mutex_unlock(&g_lock);
        return 0;
    }
    const float *remote_y = (const float *)(response + 1);
    int cursor = 0;
    for (int j = 0; j < nu; j++)
        for (int s = 0; s < S; s++)
            for (int k = 0; k < keff[s]; k++)
                if (idxs[(size_t)s * K + k] == uniq[j]) {
                    float w = weights[(size_t)s * K + k];
                    float *dst = out + (size_t)s * D;
                    const float *src = remote_y + (size_t)cursor++ * D;
                    for (int d = 0; d < D; d++) dst[d] += w * src[d];
                    break;
                }
    g_calls++;
    g_experts += (uint64_t)nu;
    g_rows += (uint64_t)total;
    g_bytes += request_bytes + sizeof(*response) + (size_t)total * D * sizeof(float);
    g_us += now_us() - started;
    pthread_mutex_unlock(&g_lock);
    return 1;
}

void coli_remote_expert_stats(FILE *stream) {
    if (!g_calls) return;
    fprintf(stream,
            "[REMOTE] %llu calls, %llu experts, %llu rows, %.2f GB, %.2f us/call, "
            "%llu fallback\n",
            (unsigned long long)g_calls, (unsigned long long)g_experts,
            (unsigned long long)g_rows, g_bytes / 1e9,
            (double)g_us / g_calls, (unsigned long long)g_fallbacks);
}

void coli_remote_expert_shutdown(void) {
    pthread_mutex_lock(&g_lock);
    Remote *r = &g_remote;
    if (r->connected) {
        ColiRemoteRequest *request = r->request;
        memset(request, 0, sizeof(*request));
        request->magic = COLI_REMOTE_MAGIC;
        request->version = COLI_REMOTE_VERSION;
        request->seq = ++g_seq;
        write_request(r, sizeof(*request), 1);
        wait_for(r, 1);
    }
    cleanup();
    pthread_mutex_unlock(&g_lock);
}
