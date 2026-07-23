#include "backend_cuda.h"
#include "remote_expert_protocol.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct PeerInfo {
    ibv_gid gid;
    uint32_t qpn, psn;
    uint64_t target_addr;
    uint32_t target_rkey;
};

struct Expert {
    ColiCudaTensor *gate{}, *up{}, *down{};
};

static void fail(const char *what) {
    std::perror(what);
    std::exit(1);
}

static void send_all(int fd, const void *data, size_t size) {
    const char *p = (const char *)data;
    while (size) {
        ssize_t n = send(fd, p, size, MSG_NOSIGNAL);
        if (n <= 0) fail("send");
        p += n;
        size -= (size_t)n;
    }
}

static void recv_all(int fd, void *data, size_t size) {
    char *p = (char *)data;
    while (size) {
        ssize_t n = recv(fd, p, size, MSG_WAITALL);
        if (n <= 0) fail("recv");
        p += n;
        size -= (size_t)n;
    }
}

static int listen_tcp(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) fail("socket");
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons((uint16_t)port);
    if (bind(fd, (sockaddr *)&address, sizeof(address)) || listen(fd, 1))
        fail("listen");
    std::fprintf(stderr, "[REMOTE] worker ready on TCP port %d\n", port);
    int client = accept(fd, nullptr, nullptr);
    if (client < 0) fail("accept");
    close(fd);
    return client;
}

struct Rdma {
    ibv_context *context{};
    ibv_pd *pd{};
    ibv_cq *cq{};
    ibv_qp *qp{};
    ibv_mr *request_mr{}, *response_mr{};
    void *request{}, *response{};
    PeerInfo self{}, peer{};
    int tcp = -1, gid_index;

    Rdma(const char *device_name, int gid, int tcp_fd) : tcp(tcp_fd), gid_index(gid) {
        int count = 0;
        ibv_device **devices = ibv_get_device_list(&count);
        ibv_device *device = nullptr;
        for (int i = 0; i < count; i++)
            if (!std::strcmp(ibv_get_device_name(devices[i]), device_name))
                device = devices[i];
        if (!device) throw std::runtime_error("RDMA device not found");
        context = ibv_open_device(device);
        ibv_free_device_list(devices);
        if (!context) fail("ibv_open_device");
        pd = ibv_alloc_pd(context);
        if (!pd) fail("ibv_alloc_pd");
        cq = ibv_create_cq(context, 16, nullptr, nullptr, 0);
        if (!cq) fail("ibv_create_cq");
        ibv_qp_init_attr init{};
        init.send_cq = init.recv_cq = cq;
        init.qp_type = IBV_QPT_RC;
        init.cap.max_send_wr = init.cap.max_recv_wr = 16;
        init.cap.max_send_sge = init.cap.max_recv_sge = 1;
        qp = ibv_create_qp(pd, &init);
        if (!qp || posix_memalign(&request, 4096, COLI_REMOTE_REQUEST_BYTES) ||
            posix_memalign(&response, 4096, COLI_REMOTE_RESPONSE_BYTES))
            fail("RDMA allocation");
        request_mr = ibv_reg_mr(pd, request, COLI_REMOTE_REQUEST_BYTES,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        response_mr = ibv_reg_mr(pd, response, COLI_REMOTE_RESPONSE_BYTES, 0);
        if (!request_mr || !response_mr ||
            ibv_query_gid(context, 1, gid_index, &self.gid))
            fail("RDMA registration");
        self.qpn = qp->qp_num;
        self.psn = (uint32_t)(getpid() * 2654435761u) & 0xffffffu;
        self.target_addr = (uintptr_t)request;
        self.target_rkey = request_mr->rkey;

        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = 1;
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                      IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            fail("QP INIT");
        send_all(tcp, &self, sizeof(self));
        recv_all(tcp, &peer, sizeof(peer));
        std::memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_1024;
        attr.dest_qp_num = peer.qpn;
        attr.rq_psn = peer.psn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = 1;
        attr.ah_attr.grh.dgid = peer.gid;
        attr.ah_attr.grh.sgid_index = gid_index;
        attr.ah_attr.grh.hop_limit = 64;
        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV |
                                      IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                                      IBV_QP_MIN_RNR_TIMER))
            fail("QP RTR");
        std::memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = self.psn;
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT |
                                      IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                      IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
            fail("QP RTS");
        char ready = 1, peer_ready = 0;
        send_all(tcp, &ready, 1);
        recv_all(tcp, &peer_ready, 1);
    }

    void post_receive(uint64_t id) {
        ibv_recv_wr wr{}, *bad = nullptr;
        wr.wr_id = id;
        if (ibv_post_recv(qp, &wr, &bad)) fail("ibv_post_recv");
    }

    void write_response(size_t size, uint64_t id) {
        ibv_sge sge{};
        sge.addr = (uintptr_t)response;
        sge.length = (uint32_t)size;
        sge.lkey = response_mr->lkey;
        ibv_send_wr wr{}, *bad = nullptr;
        wr.wr_id = id;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.imm_data = htonl((uint32_t)id);
        wr.wr.rdma.remote_addr = peer.target_addr;
        wr.wr.rdma.rkey = peer.target_rkey;
        if (ibv_post_send(qp, &wr, &bad)) fail("ibv_post_send");
    }

    void wait(uint64_t id) {
        for (;;) {
            ibv_wc completion{};
            int n = ibv_poll_cq(cq, 1, &completion);
            if (n < 0) fail("ibv_poll_cq");
            if (!n) continue;
            if (completion.status != IBV_WC_SUCCESS) {
                std::fprintf(stderr, "RDMA completion failed: %s\n",
                             ibv_wc_status_str(completion.status));
                std::exit(1);
            }
            if (completion.wr_id == id) return;
        }
    }
};

static std::vector<Expert> load_pack(const char *path, int *max_layer) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) fail("open pack");
    struct stat st{};
    if (fstat(fd, &st)) fail("stat pack");
    const uint8_t *data = (const uint8_t *)mmap(
        nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) fail("mmap pack");
    if ((size_t)st.st_size < sizeof(ColiRemotePackHeader))
        throw std::runtime_error("short pack");
    const auto *header = (const ColiRemotePackHeader *)data;
    if (std::memcmp(header->magic, COLI_REMOTE_PACK_MAGIC, 8) ||
        header->version != COLI_REMOTE_VERSION ||
        header->hidden != COLI_REMOTE_HIDDEN ||
        header->inter != COLI_REMOTE_INTER ||
        header->records_offset + (uint64_t)header->count *
            sizeof(ColiRemotePackRecord) > (uint64_t)st.st_size)
        throw std::runtime_error("invalid pack");
    const auto *records = (const ColiRemotePackRecord *)(data + header->records_offset);
    *max_layer = 0;
    for (uint32_t i = 0; i < header->count; i++)
        if (records[i].layer > *max_layer) *max_layer = records[i].layer;
    std::vector<Expert> experts((size_t)(*max_layer + 1) * 256);
    int device = 0;
    if (!coli_cuda_init(&device, 1)) throw std::runtime_error("CUDA init");
    for (uint32_t i = 0; i < header->count; i++) {
        const auto &r = records[i];
        for (int j = 0; j < 6; j++)
            if (r.offsets[j] + r.sizes[j] > (uint64_t)st.st_size)
                throw std::runtime_error("pack tensor out of range");
        Expert &e = experts[(size_t)r.layer * 256 + r.eid];
        if (!coli_cuda_tensor_upload(&e.gate, data + r.offsets[0],
                                     (const float *)(data + r.offsets[1]),
                                     2, COLI_REMOTE_HIDDEN, COLI_REMOTE_INTER, 0) ||
            !coli_cuda_tensor_upload(&e.up, data + r.offsets[2],
                                     (const float *)(data + r.offsets[3]),
                                     2, COLI_REMOTE_HIDDEN, COLI_REMOTE_INTER, 0) ||
            !coli_cuda_tensor_upload(&e.down, data + r.offsets[4],
                                     (const float *)(data + r.offsets[5]),
                                     2, COLI_REMOTE_INTER, COLI_REMOTE_HIDDEN, 0))
            throw std::runtime_error("CUDA expert upload");
        if ((i + 1) % 256 == 0)
            std::fprintf(stderr, "[REMOTE] loaded %u/%u experts\n", i + 1, header->count);
    }
    uint32_t count = header->count;
    munmap((void *)data, (size_t)st.st_size);
    close(fd);
    std::fprintf(stderr, "[REMOTE] loaded %u experts from %s\n", count, path);
    return experts;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr,
                     "usage: %s PACK IB_DEVICE GID_INDEX [TCP_PORT]\n", argv[0]);
        return 2;
    }
    int max_layer = 0;
    auto experts = load_pack(argv[1], &max_layer);
    int tcp = listen_tcp(argc == 5 ? std::atoi(argv[4]) : 19066);
    Rdma rdma(argv[2], std::atoi(argv[3]), tcp);
    std::vector<float> packed((size_t)COLI_REMOTE_MAX_ROWS * COLI_REMOTE_HIDDEN);
    uint64_t calls = 0, rows = 0;
    double compute_us = 0;
    rdma.post_receive(1);
    for (;;) {
        rdma.wait(1);
        const auto *request = (const ColiRemoteRequest *)rdma.request;
        if (request->magic != COLI_REMOTE_MAGIC ||
            request->version != COLI_REMOTE_VERSION) {
            std::fprintf(stderr, "[REMOTE] invalid request header\n");
            return 1;
        }
        if (!request->count) break;
        auto *response = (ColiRemoteResponse *)rdma.response;
        response->magic = COLI_REMOTE_MAGIC;
        response->version = COLI_REMOTE_VERSION;
        response->seq = request->seq;
        response->status = 0;
        response->count = request->count;
        response->total_rows = request->total_rows;
        if (request->S < 1 || request->S > COLI_REMOTE_MAX_S ||
            request->count > COLI_REMOTE_MAX_EXPERTS ||
            request->total_rows > COLI_REMOTE_MAX_ROWS ||
            request->layer > (uint32_t)max_layer) {
            response->status = 1;
        }
        ColiCudaTensor *gates[COLI_REMOTE_MAX_EXPERTS];
        ColiCudaTensor *ups[COLI_REMOTE_MAX_EXPERTS];
        ColiCudaTensor *downs[COLI_REMOTE_MAX_EXPERTS];
        int group_rows[COLI_REMOTE_MAX_EXPERTS];
        int total = 0;
        const float *request_x = (const float *)(request + 1);
        if (!response->status)
            for (uint32_t i = 0; i < request->count; i++) {
                int eid = request->eids[i];
                int nr = request->rows[i];
                if (eid < 0 || eid >= 256 || nr < 1 ||
                    total + nr > (int)request->total_rows) {
                    response->status = 2;
                    break;
                }
                Expert &e = experts[(size_t)request->layer * 256 + eid];
                if (!e.gate || !e.up || !e.down) {
                    response->status = 3;
                    break;
                }
                gates[i] = e.gate;
                ups[i] = e.up;
                downs[i] = e.down;
                group_rows[i] = nr;
                for (int r = 0; r < nr; r++) {
                    int token = request->tokrow[total];
                    if (token < 0 || token >= (int)request->S) {
                        response->status = 4;
                        break;
                    }
                    std::memcpy(packed.data() + (size_t)total * COLI_REMOTE_HIDDEN,
                                request_x + (size_t)token * COLI_REMOTE_HIDDEN,
                                COLI_REMOTE_HIDDEN * sizeof(float));
                    total++;
                }
                if (response->status) break;
            }
        if (!response->status && total != (int)request->total_rows)
            response->status = 5;
        if (!response->status) {
            auto start = std::chrono::steady_clock::now();
            if (!coli_cuda_expert_group(
                    gates, ups, downs, group_rows, (int)request->count,
                    (float *)(response + 1), packed.data(), nullptr, nullptr,
                    0, 0, 0, nullptr))
                response->status = 6;
            auto end = std::chrono::steady_clock::now();
            compute_us +=
                std::chrono::duration<double, std::micro>(end - start).count();
        }
        uint32_t safe_rows = response->status ? 0 : request->total_rows;
        response->total_rows = safe_rows;
        calls++;
        rows += safe_rows;
        rdma.post_receive(1);
        size_t bytes = sizeof(*response) +
                       (size_t)safe_rows * COLI_REMOTE_HIDDEN * sizeof(float);
        rdma.write_response(bytes, 2);
        rdma.wait(2);
    }
    std::fprintf(stderr,
                 "[REMOTE] stopped after %llu calls, %llu rows, %.2f us compute/call\n",
                 (unsigned long long)calls, (unsigned long long)rows,
                 calls ? compute_us / calls : 0);
    return 0;
}
