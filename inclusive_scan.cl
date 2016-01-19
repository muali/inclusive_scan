#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(
		uint size,
		__global float * input,
		__global float * output, 
		__local float * a, 
		__local float * b,
		__global float * reduce_output)
{
    uint gid = get_global_id(0);
	uint group_id = get_group_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

	if (gid < size) 
		a[lid] = b[lid] = input[gid];

    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(lid > (s-1))
        {
            b[lid] = a[lid] + a[lid-s];
        }
        else
        {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
	if (gid < size)
	{
		output[gid] = a[lid];
		if (lid == block_size - 1)
			reduce_output[group_id] = a[lid];
	}

}

__kernel void back_stage(
		uint size,
		__global float * input,
		__global float * output,
		__global float * reduced)
{
	uint gid = get_global_id(0);
	uint group_id = get_group_id(0);

	if (gid >= size) return;

	float res = input[gid];
	if (group_id > 0)
		res += reduced[group_id - 1];
	output[gid] = res;
}