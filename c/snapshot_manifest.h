#ifndef COLI_SNAPSHOT_MANIFEST_H
#define COLI_SNAPSHOT_MANIFEST_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "tensor_format.h"

#define COLI_MANIFEST_FILE "colibri-manifest.json"
#define COLI_CUTLASS_451_REV "2e602843e75100d0e03934efb386b3e1e35d7907"

typedef struct {
    int present, version, resident_format, expert_format, group_size;
    int scale_layout, record_alignment, component_alignment;
    char repository[256], revision[128];
} ColiSnapshotManifest;

static inline const char *coli_manifest_string(jval *object, const char *key) {
    jval *value = object && object->t == J_OBJ ? json_get(object, key) : NULL;
    return value && value->t == J_STR ? value->str : NULL;
}

static inline int coli_manifest_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
    return -1;
}

/* Returns 0 for a legacy snapshot, 1 for a valid v1 manifest, -1 on error. */
static inline int coli_manifest_load(const char *snapshot, ColiSnapshotManifest *out,
                                     char *error, size_t error_cap) {
    memset(out, 0, sizeof(*out));
    char path[2304];
    if (snprintf(path, sizeof(path), "%s/%s", snapshot, COLI_MANIFEST_FILE) >= (int)sizeof(path))
        return coli_manifest_error(error,error_cap,"snapshot manifest path is too long");
    FILE *file = fopen(path, "rb");
    if (!file) return errno == ENOENT ? 0 : coli_manifest_error(error,error_cap,"cannot open snapshot manifest");
    if (fseek(file,0,SEEK_END) || ftell(file)<0) { fclose(file); return coli_manifest_error(error,error_cap,"cannot size snapshot manifest"); }
    long size=ftell(file);
    if (size <= 0 || size > (1<<20) || fseek(file,0,SEEK_SET)) { fclose(file); return coli_manifest_error(error,error_cap,"snapshot manifest has invalid size"); }
    char *text=(char*)malloc((size_t)size+1);
    if (!text) { fclose(file); return coli_manifest_error(error,error_cap,"out of memory reading snapshot manifest"); }
    size_t got=fread(text,1,(size_t)size,file); fclose(file); text[got]=0;
    if (got!=(size_t)size) { free(text); return coli_manifest_error(error,error_cap,"short read of snapshot manifest"); }
    char *arena=NULL; jval *root=json_parse(text,&arena); free(text);
    if (!root || root->t!=J_OBJ) { json_free(root); free(arena); return coli_manifest_error(error,error_cap,"snapshot manifest is not a JSON object"); }
    const char *schema=coli_manifest_string(root,"schema"); jval *version=json_get(root,"version");
    if (!schema || strcmp(schema,"colibri.snapshot") || !version || version->t!=J_NUM || version->num!=1) {
        json_free(root); free(arena); return coli_manifest_error(error,error_cap,"unsupported snapshot manifest schema/version"); }
    jval *source=json_get(root,"source");
    const char *repo=coli_manifest_string(source,"repository"), *rev=coli_manifest_string(source,"revision");
    if (!repo || !*repo || !rev || !*rev || strlen(repo)>=sizeof(out->repository) || strlen(rev)>=sizeof(out->revision)) {
        json_free(root); free(arena); return coli_manifest_error(error,error_cap,"manifest requires bounded source repository and exact revision");
    }
    const char *resident=coli_manifest_string(root,"resident_precision");
    out->resident_format=resident&&!strcmp(resident,"bf16") ? COLI_TENSOR_BF16
                         : resident&&!strcmp(resident,"int8-row") ? COLI_TENSOR_INT8_ROW : -1;
    jval *expert=json_get(root,"routed_experts");
    const char *format=coli_manifest_string(expert,"format");
    const char *weight_layout=coli_manifest_string(expert,"weight_layout");
    const char *source_layout=coli_manifest_string(expert,"source_scale_layout");
    const char *scale_layout=coli_manifest_string(expert,"scale_layout");
    const char *scale_dtype=coli_manifest_string(expert,"scale_dtype");
    const char *tensor_dtype=coli_manifest_string(expert,"tensor_scale_dtype");
    const char *input_dtype=coli_manifest_string(expert,"input_scale_dtype");
    jval *group=expert&&expert->t==J_OBJ?json_get(expert,"group_size"):NULL;
    if (out->resident_format<0 || !format || strcmp(format,"modelopt-nvfp4-e2m1") ||
        !group || group->t!=J_NUM || group->num!=16 ||
        !weight_layout || strcmp(weight_layout,"e2m1-low-nibble-even") ||
        !source_layout || strcmp(source_layout,"modelopt-row-major-o-by-ceil-i16") ||
        !scale_layout || strcmp(scale_layout,"cutlass-sm1xx-sf-atom-128x4-v1") ||
        !scale_dtype || strcmp(scale_dtype,"fp8-e4m3fn") ||
        !tensor_dtype || strcmp(tensor_dtype,"f32") || !input_dtype || strcmp(input_dtype,"f32")) {
        json_free(root); free(arena); return coli_manifest_error(error,error_cap,"unsupported native NVFP4 tensor metadata");
    }
    jval *cutlass=json_get(root,"cutlass");
    const char *cv=coli_manifest_string(cutlass,"version"), *cr=coli_manifest_string(cutlass,"revision");
    if (!cv || strcmp(cv,"4.5.1") || !cr || strcmp(cr,COLI_CUTLASS_451_REV)) {
        json_free(root); free(arena); return coli_manifest_error(error,error_cap,"manifest does not pin supported CUTLASS 4.5.1 revision");
    }
    jval *record=json_get(root,"expert_record"), *alignment=record&&record->t==J_OBJ?json_get(record,"alignment"):NULL;
    jval *immutable=record&&record->t==J_OBJ?json_get(record,"immutable"):NULL;
    jval *addressable=record&&record->t==J_OBJ?json_get(record,"independently_addressable"):NULL;
    jval *component=record&&record->t==J_OBJ?json_get(record,"component_alignment"):NULL;
    const char *record_layout=coli_manifest_string(record,"layout");
    if (!alignment || alignment->t!=J_NUM || (alignment->num!=4096 && alignment->num!=16384)) {
        json_free(root); free(arena); return coli_manifest_error(error,error_cap,"unsupported expert record alignment");
    }
    if(!immutable||immutable->t!=J_BOOL||!immutable->boolean||
       !addressable||addressable->t!=J_BOOL||!addressable->boolean){
        json_free(root);free(arena);return coli_manifest_error(error,error_cap,"expert records must be immutable and independently addressable");
    }
    out->component_alignment=1; /* v1 payloads require compatibility staging */
    if(component){
        if(component->t!=J_NUM||component->num!=16||!record_layout||
           strcmp(record_layout,"component-aligned-v2")){
            json_free(root);free(arena);return coli_manifest_error(error,error_cap,
                "unsupported expert component alignment/layout");
        }
        out->component_alignment=16;
    }else if(record_layout){
        json_free(root);free(arena);return coli_manifest_error(error,error_cap,
            "expert record layout requires component alignment");
    }
    out->present=1; out->version=1; out->expert_format=COLI_TENSOR_MODELOPT_NVFP4;
    out->group_size=16; out->scale_layout=COLI_SCALE_CUTLASS_SM1XX_128X4;
    out->record_alignment=(int)alignment->num;
    snprintf(out->repository,sizeof(out->repository),"%s",repo);
    snprintf(out->revision,sizeof(out->revision),"%s",rev);
    json_free(root); free(arena); return 1;
}

#endif
