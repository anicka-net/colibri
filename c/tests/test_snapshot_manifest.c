#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../snapshot_manifest.h"

static int write_manifest(const char *dir, const char *group) {
    char path[512]; snprintf(path,sizeof(path),"%s/%s",dir,COLI_MANIFEST_FILE);
    FILE *f=fopen(path,"wb"); if(!f)return 0;
    fprintf(f,"{\"schema\":\"colibri.snapshot\",\"version\":1,"
        "\"source\":{\"repository\":\"org/model\",\"revision\":\"abc123\"},"
        "\"resident_precision\":\"bf16\",\"routed_experts\":{"
        "\"format\":\"modelopt-nvfp4-e2m1\",\"group_size\":%s,"
        "\"weight_layout\":\"e2m1-low-nibble-even\","
        "\"source_scale_layout\":\"modelopt-row-major-o-by-ceil-i16\","
        "\"scale_layout\":\"cutlass-sm1xx-sf-atom-128x4-v1\","
        "\"scale_dtype\":\"fp8-e4m3fn\",\"tensor_scale_dtype\":\"f32\","
        "\"input_scale_dtype\":\"f32\"},\"expert_record\":{\"alignment\":4096,"
        "\"immutable\":true,\"independently_addressable\":true},"
        "\"cutlass\":{\"version\":\"4.5.1\",\"revision\":\"%s\"}}",
        group,COLI_CUTLASS_451_REV);
    return fclose(f)==0;
}

int main(void) {
    char dir[]="/tmp/coli-manifest-XXXXXX", error[256];
    if(!mkdtemp(dir))return 1;
    ColiSnapshotManifest m;
    if(coli_manifest_load(dir,&m,error,sizeof(error))!=0)return 2;
    if(!write_manifest(dir,"16"))return 3;
    if(coli_manifest_load(dir,&m,error,sizeof(error))!=1 || !m.present ||
       m.expert_format!=COLI_TENSOR_MODELOPT_NVFP4 || m.resident_format!=COLI_TENSOR_BF16 ||
       strcmp(m.repository,"org/model") || strcmp(m.revision,"abc123"))return 4;
    if(!write_manifest(dir,"32"))return 5;
    if(coli_manifest_load(dir,&m,error,sizeof(error))!=-1 || !strstr(error,"NVFP4"))return 6;
    char path[512];snprintf(path,sizeof(path),"%s/%s",dir,COLI_MANIFEST_FILE);unlink(path);rmdir(dir);
    puts("snapshot manifest: ok"); return 0;
}
