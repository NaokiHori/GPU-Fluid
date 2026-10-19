#include "compute_pipeline.h"

int compute_pipeline_initialize(
    id<MTLDevice> device,
    const char * const shader_path,
    const char * const function_name,
    NSDictionary<NSString *, id> * const macros,
    compute_pipeline_t * const compute_pipeline
) {
  NSCAssert(macros != nil, @"Macros dictionary must not be nil.");
  NSError * error = nil;
  NSString * source_str = [
    NSString stringWithContentsOfFile:[NSString stringWithUTF8String:shader_path]
    encoding:NSUTF8StringEncoding
    error:&error
  ];
  if (nil == source_str) {
    NSLog(@"[Error] Failed to load shader file '%s': %@", shader_path, [error localizedDescription]);
    return 1;
  }
  MTLCompileOptions * const options = [MTLCompileOptions new];
  options.preprocessorMacros = macros;
  id<MTLLibrary> lib = [device newLibraryWithSource:source_str options:options error:&error];
  if (nil == lib) {
    NSLog(@"[Shader Error] %@", [error localizedDescription]);
    return 1;
  }
  id<MTLFunction> func = [lib newFunctionWithName:[NSString stringWithUTF8String:function_name]];
  if (nil == func) {
    NSLog(@"[Function Error] Cannot find function: %s", function_name);
    return 1;
  }
  compute_pipeline->pipeline_state = [device newComputePipelineStateWithFunction:func error:&error];
  if (nil == compute_pipeline->pipeline_state) {
    NSLog(@"[Pipeline Error] %@", [error localizedDescription]);
    return 1;
  }
  return 0;
}

