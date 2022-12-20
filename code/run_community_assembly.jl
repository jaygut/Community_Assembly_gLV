#!/data/gent/435/vsc43582/Software/julia-1.5.2/bin/julia

repl_num = popfirst!(ARGS); #1, ..., N
net_type = popfirst!(ARGS); #compet_only | compet_mutual

include("community_assembly_v0.jl")

init_size,target_size = (5, 50)
sample_range_A = (-1.,-1e-2)
edge_connect_probab = 1.0 #rand(Uniform(0.5,1))
set_intrasps_effects=-1
prob_scale=0.25
perturb_intrinsic_grs=false
iter_limit=2000
A_symmetric=true
dyn_var_thr = 1e-4

if(net_type=="compet_only")

    output_folder = "../output/compet_commun_assembly_full_record/";

    dt = @elapsed begin
        CPUdt = @CPUelapsed runAdaptRadiationCommunAssembly(init_size,target_size, 
                                                            sample_range_A = sample_range_A,
                                                            edge_connect_probab = edge_connect_probab,
                                                            prob_scale=prob_scale,
                                                            set_intrasps_effects=set_intrasps_effects,
                                                            perturb_intrinsic_grs=perturb_intrinsic_grs,
                                                            iter_limit=iter_limit, 
                                                            A_symmetric=A_symmetric,
                                                            dyn_var_thr=dyn_var_thr,
                                                            repl=repl_num,
                                                            output_folder=output_folder);
    
    end    

else

    output_folder = "../output/compet_mutual_commun_assembly_full_record/";

    dt = @elapsed begin
        CPUdt = @CPUelapsed runAdaptRadiationCommunAssembly2(init_size,target_size, 
                                                             sample_range_A = sample_range_A,
                                                             edge_connect_probab = edge_connect_probab,
                                                             prob_scale=prob_scale,
                                                             set_intrasps_effects=set_intrasps_effects,
                                                             perturb_intrinsic_grs=perturb_intrinsic_grs,
                                                             iter_limit=iter_limit, 
                                                             A_symmetric=A_symmetric,
                                                             add_mutual_inters = true,
                                                             dyn_var_thr=dyn_var_thr,
                                                             repl=repl_num,
                                                             output_folder=output_folder);
    end
    
end
println("Elapsed time for community assembly run, replicate $repl_num, was: $dt");

