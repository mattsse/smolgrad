use ocl::ProQue;

fn unop(code: impl AsRef<str>) -> String {
    format!(
        "__kernel void unop(__global const float *a_g, __global float *res_g) {{
    int gid = get_global_id(0);
    float a = a_g[gid];
    res_g[gid] = {};
  }}",
        code.as_ref()
    )
}

fn trivial() -> ocl::Result<()> {
    let _src = r#"
        __kernel void add(__global float* buffer, float s) {
            buffer[get_global_id(0)] += s;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(unop("max(a, (float)0.)"))
        .dims(10)
        .build()?;

    let vec_out = vec![50.0f32; 10];

    let output = pro_que
        .buffer_builder::<f32>()
        .copy_host_slice(&vec_out)
        .build()?;

    let input = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("unop")
        .arg(&output)
        .arg(&input)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut vec_out = vec![0.0f32; output.len()];
    let mut vec_in = vec![0.0f32; output.len()];

    output.read(&mut vec_out).enq()?;
    input.read(&mut vec_in).enq()?;

    // println!("The value at index [{}] is now '{}'!", 9, vec[9]);
    dbg!(vec_out);
    dbg!(vec_in);
    Ok(())
}

fn main() {
    trivial().unwrap();
}
