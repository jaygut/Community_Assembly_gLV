using DifferentialEquations, DiffEqBiological, ModelingToolkit, Latexify
using Random, LinearAlgebra, Distributions, StatsBase, Statistics, HypothesisTests, Combinatorics
using HDF5, JLD, DelimitedFiles, DataFrames, CSV, CPUTime, Queryverse
using LightGraphs, GraphPlot, SparseArrays, Plots, EcologicalNetworks, EcologicalNetworksPlots, GraphRecipes, MetaGraphs, Compose

#Utilities to simulate community assembly under the classical gLV modeling approach
#Main function for community assembly runAdaptRadiationCommunAssembly applies an 
#Adaptive Radiation scheme, which is embedded into an MCMC-like algorithm to 
#build progrssively a large community via invasion events of one species at a time

#Define a type to hold ODEs system solution
mutable struct ODEsSystSolStruct
    numsol::Any
    paramstruct::Dict{String,Any}
end


function calcSumPairwiseSpsInteractions(u, i, params)
    "Use function to calculate the dot product between row-vectors in the connectivity matrix and species"
#     return sum(u.*params['A'][i, :])
    return dot(u, params["connectMatrix"][i, :])
end


function gLV!(du, u, params, t)
    "Definition of a conventional generalized Lotka-Volterra network as a system of ODEs"
    for i in 1:length(u)
        du[i] = u[i] * (params["growthRates"][i] + calcSumPairwiseSpsInteractions(u, i, params))
    end
end


function solvegLV(params::Dict{String,Any}, limit_val::Float64=10., save_at::Float64=1.0)
    
    "
    Use function to numerically solve a gLV system. This function implements a callback 
    to check whether computed values remain within a range, and if so terminates the integration. 
    This is required to avoid spending too much time integrating stiff parameter settings. 
    Values are constrained to the range [0, limit_val]
    
    Parameter settings (params) is cast as a Dict with the following keys:
     - connectMatrix
     - growthRates
     - initConds
     - endTimePoint
    
    NOTE: here we use the function gLV as defined previously
    "
    u0 = params["initConds"]
    tspan = (0.0, params["endTimePoint"])
    prob = ODEProblem(gLV!, u0, tspan, params);
    
    condition(u,t,integrator) = any(u .< 0.0) || any(u .> limit_val)
#     condition(u,t,integrator) = any(map(x->any(x .> 10),eachcol(DataFrame(sol.u))).==1)
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition,affect!)
    sol = solve(prob,
                AutoTsit5(Rosenbrock23(autodiff=false)),#Rosenbrock23(autodiff=false),#AutoVern7(Rodas5()),
                callback=cb,
                alg_hints=[:stiff],
                save_at=save_at)
    return sol
end

function generateErdosRenyiNetwork(netsize::Int64=16,
                                   probab::Float64=0.25,
                                   nselfloops::Int64=10,
                                   perturb_range::Tuple{Float64,Float64}=(-10.0,10.0)
                                   )::Array{Float64,2}
   
    "Generate an Erdos-Renyi random graph, with a given number of self-loops if requested.
     The entries in the matrix are then assigned randomly chosen weights (perturb_range).
     Use the resulting matrix to simulate a conventional gLV model."
 
    mask_array = zeros(netsize,netsize);
    er_rnd_g = LightGraphs.SimpleGraphs.erdos_renyi(netsize,probab,is_directed=true);
    adj_mat = LightGraphs.adjacency_matrix(er_rnd_g);
    row, col, vals = SparseArrays.findnz(adj_mat);
    #Thes values must be set to −10 < αij < 10
    for (r,c) in zip(row,col)
        mask_array[r,c] = rand(Uniform(perturb_range...));
    end

    #Self-interactions are set to the lower-bound value (intraspecies competition), to prevent density from blowing up!
    if nselfloops>0
        slidx = sample(1:netsize,nselfloops,replace=false);
        for i in slidx
#             lbw = perturb_range[1];
#             mask_array[i,i] = rand(Uniform(lbw,lbw/2));
            mask_array[i,i] = perturb_range[1]
        end
    end

    return mask_array
    
end

function assembleParamStruct(connect_mat::Array{Float64,2}, 
                             growth_rates::Array{Float64,1}, 
                             ics::Array{Float64,1}, 
                             etp::Int64)::Dict{String,Any}

    "Use function to create a Dict grouping parameters to solve nuemrically the gLV model"
    keynames = ("connectMatrix", "growthRates", "initConds", "endTimePoint")
    vals = (connect_mat, growth_rates, ics, etp)
    return Dict(zip(keynames,vals))
end

function checkSol4DynSyst(param_struct::Dict{String,Any},limit_val::Float64=10.0,save_at::Float64=1.0)
    
    "Simple function to obtain numerical solution of a given species interaction network model. Use this function as
      interface to call other network models ..."
    #Solve ODEs
    sol = solvegLV(param_struct,limit_val,save_at);

end


function testTimeSeries4Stationarity(sol; tw=1:500, deterministic=:constant, lag=0, α=0.05)

    "Use the ADF test to assess time series for stationarity at a predefined significance level α = 0.05"
    ts_df = DataFrame(convert(Array,DataFrame([sol(i) for i in tw]))');
    test4all = all(convert(Array,mapcols(col -> pvalue(ADFTest(col,deterministic,lag)), ts_df) .< 0.05));
    return test4all
end


function add_mutual_ints2compet_net(num_species, init_net; 
                                    add_mutualism_perc = 0.2, 
                                    mutual_strength_range = [1e-3,1e0])
   "Use function to set a % of pairwise interactions in an initially fully competitive network to be mutualistic"
    #Set all possible pairwise interactions
    all_pairs = collect(combinations(1:num_species,2))
    #Get number of pairs involved in mutualistic interactions
    num_pair_mutual = Integer(round(add_mutualism_perc*((num_species*(num_species-1))/2)))

    #Choose pairs to set as mutualistic interacting species in the network
    mutual_ids = sample(1:length(all_pairs),num_pair_mutual,replace=false)
    mutual_pairs = all_pairs[mutual_ids]
    for mp in mutual_pairs
        set_mutual_val = rand(Uniform(mutual_strength_range...))
        init_net[mp...] = set_mutual_val
        init_net[reverse(mp)...] = set_mutual_val
    end
        return init_net
end


function rndGenerationInitSpsNetwork(num_species; 
                                     sample_range_A = (-1.,-1e-3),
                                     sample_range_grs = (1e-1,1e0),
                                     sample_range_ics = (1e-1,1e0),
                                     edge_connect_probab = rand(Uniform(0.95,1.0)),
                                     integ_steps = 2000, ub_density = 1e1, 
                                     lb_density = 1e-7, tw=1:2000, 
                                     deterministic=:constant, lag=0, α=0.05,
                                     A_symmetric = true,
                                     add_mutual_inters = false,
                                     add_mutualism_perc = 0.2, 
                                     mutual_strength_range = [1e-3,1e0])

    "Use function to generate a species interaction network with stationary abundance profiles."
    test4feasib = true
    #n=0;
    while test4feasib  

        ics = rand(Uniform(0.0,1.0),num_species);
        grs = ones(num_species);

        #Create randomly wired graph to initialize search space algorithm   
        erdos_renyi_network_matrix = generateErdosRenyiNetwork(num_species, edge_connect_probab, 
                                                               num_species, sample_range_A);
        #Set matrix to be symmetric if required
        if(A_symmetric)
            erdos_renyi_network_matrix = convert(Array,Symmetric(erdos_renyi_network_matrix));
        end
        
        #Set some of the species pairs to be mutualistic (initially with symmetrical interaction strength)
        if(add_mutual_inters)
            erdos_renyi_network_matrix = add_mutual_ints2compet_net(num_species,erdos_renyi_network_matrix; 
                                                                    add_mutualism_perc = add_mutualism_perc, 
                                                                    mutual_strength_range = mutual_strength_range);
        end
        rnd_param_struct = assembleParamStruct(erdos_renyi_network_matrix,grs,ics,integ_steps);
        sol = checkSol4DynSyst(rnd_param_struct,ub_density);

        cond1 = (sol.retcode == :Success) #Checking for successful integration until end time point
        cond2 = (all(round.(sol.u[end],digits=6) .> lb_density)) #Checking that all species abundances at end time point > 0 
        cond3 = testTimeSeries4Stationarity(sol; tw=tw, deterministic=deterministic, lag=lag, α=α); #Checking for stationary abundances

        if(cond1 & cond2 & cond3)
            test4feasib = false
            return rnd_param_struct,sol;
        else
            test4feasib = true
        end
        #n += 1;
    end

end


function communAssemblyAdaptRadiationScheme(ref_param_struct; 
                                            μ_mutat = 1, σ_mutat = 0.1, 
                                            clip_lo=-1, clip_hi=0,
                                            clip2_lo=0, clip2_hi=1,
                                            gr_clip_lo = -2.0, gr_clip_hi = -1e-3,
                                            invader_init_abund=1e-5,
                                            ub_density = 1e1,
                                            set_intrasps_effects=-1,
                                            perturb_intrinsic_grs=false)

    "Use function to create a new parameter structure that incorporates feats for a candidate invader. The community
     assembly scheme used is referred to as adaptive radiation scheme in the paper:
     Network spandrels reflect ecological assembly (see description at the top of the jupyter notebook: AssemblyDynsEcolNets_v0.ipynb)"

    #Dict to store augmented parameters
    augmented_param_struct = Dict{String,Any}();

    #Resident community's stationary densities before invasion
    net_dyns_sol = checkSol4DynSyst(ref_param_struct,ub_density);
    res_ss_abunds_pre_invasion = net_dyns_sol(ref_param_struct["endTimePoint"]);
    
    cmtx_copy = copy(ref_param_struct["connectMatrix"]);
    ics = copy(res_ss_abunds_pre_invasion);
    #Add initial abundance for the tested invader: low abundance
    append!(ics,invader_init_abund)
    #No. sps in the community
    net_size = size(cmtx_copy,2);
    
    #Chose a species (col) to duplicate and then perturb
    sps2dupl_id = sample(1:net_size);
    #Add intrinsic growth to invader
    grs = copy(ref_param_struct["growthRates"]);
    if(!perturb_intrinsic_grs)#Keep intrinsic growth rates constant throughout community assembly process      
        append!(grs,grs[sps2dupl_id])
    #Otherwise, allow intrinsic growth rates to mutate
    else
        new_gr = grs[sps2dupl_id]*rand(Normal(μ_mutat,σ_mutat));
        append!(grs,clamp(new_gr, gr_clip_lo, gr_clip_hi));
    end
    
    #mutate outgoing links, which represent the effects of the duplicated sps on the other species
    mut_outgoing_effects = cmtx_copy[:,sps2dupl_id] .* rand(Normal(μ_mutat,σ_mutat),net_size);
    #mutate incoming links, which represent the effects of the other species on the duplicated sps
    mut_incoming_effects = cmtx_copy[sps2dupl_id,:] .* rand(Normal(μ_mutat,σ_mutat),net_size);
    #Keep values bounded both for competitive and for mutualistic interactions
    mut_outgoing_effects = [x < 0 ? clamp(x,clip_lo, clip_hi) : clamp(x,clip2_lo, clip2_hi) for x in mut_outgoing_effects]
    mut_incoming_effects = [x < 0 ? clamp(x,clip_lo, clip_hi) : clamp(x,clip2_lo, clip2_hi) for x in mut_incoming_effects]
#     mut_outgoing_effects = clamp.(mut_outgoing_effects, clip_lo, clip_hi);
#     mut_incoming_effects = clamp.(mut_incoming_effects, clip_lo, clip_hi);

    #Create augmented undefined matrix template to fill in with update entries 
    augmented_mtx = Array{Float64}(undef, net_size+1,net_size+1);
    #Fill augmented matrix with entries from the original matrix
    augmented_mtx[1:net_size,1:net_size] .=  cmtx_copy;
    #Fill new column and row with updated/mutated values
    augmented_mtx[1:net_size,end] = mut_outgoing_effects;
    augmented_mtx[end,1:net_size] = mut_incoming_effects;
    #Keep diagonal values to pre-specified values indicating e.g. intraspecific competition
    augmented_mtx[diagind(augmented_mtx)] .= set_intrasps_effects;
    
    #Fill new dict
    augmented_param_struct["initConds"] = ics;
    augmented_param_struct["growthRates"] = grs;
    augmented_param_struct["connectMatrix"] = augmented_mtx;
    augmented_param_struct["endTimePoint"] = ref_param_struct["endTimePoint"];
    
    #Intrinsic growth rate of candidate invader
    gr_invader = augmented_param_struct["growthRates"][end];
    #Incoming effects on candidate invader
    incoming_effects_invader = augmented_param_struct["connectMatrix"][end,1:end-1];
    #Compute invasion criterion: invader growth rate when rare, which describes 
    #short-term dynamics immediately after the species’ introduction
    invasion_score = gr_invader + sum(incoming_effects_invader .* res_ss_abunds_pre_invasion);    

    return augmented_param_struct,invasion_score,sps2dupl_id
    
end


function addNewSps2ResidentCommunity(ref_param_struct; ub_density=1e1, 
                                     deterministic=:constant, lag=0, α=0.05,
                                     inv_abund_min_thr=1e-1, set_intrasps_effects=-1,
                                     perturb_intrinsic_grs=false)

    "Use function to add a newly created species to a resident community (encoded in ref_param_struct), and
    find the right set of parameter combinations (pairwise sps interactions) for which the following criteria 
    are met: 
            -1) invasibility of the newly created species (using the classical invasion criterion)
            -2) stationarity abundance profiles 
            -3) feasibility
    The output of this function can be used for further invasion (community assembly experiments) once near-zero
    abundance species are removed from the newly created community
    "

    let augmented_param_struct,sol_invaded_community,ss_abundances;
        tw=1:ref_param_struct["endTimePoint"]; 
    
        failed_invasion_criterion = true
        
        while (failed_invasion_criterion)
            #Check that the new species can effectively invade the resident community
            augmented_param_struct, invasion_score, sps2dupl_id = communAssemblyAdaptRadiationScheme(ref_param_struct,
                                                                                                     set_intrasps_effects=set_intrasps_effects,
                                                                                                     perturb_intrinsic_grs=perturb_intrinsic_grs);
            #Assess dynamics of invaded community
            sol_invaded_community = checkSol4DynSyst(augmented_param_struct,ub_density);
            #Check for feasibility: steady state sps abundances > 0
            ss_abundances = sol_invaded_community(augmented_param_struct["endTimePoint"]);
            feasib_criterion = all(ss_abundances .> 0);
            #Check for stationarity using the ADF test
            ss_criterion = testTimeSeries4Stationarity(sol_invaded_community; tw=tw, 
                                                       deterministic=deterministic, lag=lag, α=α);
            #Summary: checking for invasibility, feasibility, steady state, and also making sure that
            #the steady state density of the newly added species > thr
            if (invasion_score > 0 && feasib_criterion && ss_criterion && ss_abundances[end]>inv_abund_min_thr)
                failed_invasion_criterion = false
            else
                failed_invasion_criterion = true
            end
        end

        return augmented_param_struct, sol_invaded_community, ss_abundances
        
    end
end


function pruningCommunity4ExtinctSps(param_struct,ss_abundances;abund_min_thr=1e-6)

    "Remove any species that was driven to fixation aftern an invasion event"
    #Check if any species in the community went extinct following the invasion of the new species
    ext_sps_ids = findall(ss_abundances.<abund_min_thr);
    if(~isempty(ext_sps_ids))
        #Species IDs that are above the abund_min_thr
        sps_ids2keep = sort(collect(setdiff(Set(1:length(ss_abundances)),Set(ext_sps_ids))));    
        #Pruning down community by removing the corresponding parameters
        param_struct["connectMatrix"] = param_struct["connectMatrix"][sps_ids2keep, sps_ids2keep];
        param_struct["growthRates"] = param_struct["growthRates"][sps_ids2keep];
        param_struct["initConds"] = param_struct["initConds"][sps_ids2keep];   
    end
    
    return param_struct   
end

function checkingNewlyAssembledCommunity(augmented_param_struct,ss_abundances; deterministic=:constant, lag=0, α=0.05)

    "Checking that newly assembled community with invader and removed sps met the criteria of feasibility and stationarity"
    
    #Check whether any sps in the community went extinct and remove them
    filtered_param_struct = pruningCommunity4ExtinctSps(augmented_param_struct,ss_abundances;abund_min_thr=1e-6);
    #Set conditions met by default, but double check below by default
    conds_met = true

    #Check if any sps was removed from community
    if(length(filtered_param_struct["growthRates"]) != length(augmented_param_struct["growthRates"]))
        #Double check that the resulting community meets the imposed criteria
        sol_filtered_community = checkSol4DynSyst(filtered_param_struct,ub_density);
        #Check for conditions
        ss_abundances = sol_filtered_community(filtered_param_struct["endTimePoint"]);
        feasib_criterion = all(ss_abundances .> 0);
        #Check for stationarity using the ADF test
        tw=1:filtered_param_struct["endTimePoint"];
        ss_criterion = testTimeSeries4Stationarity(sol_filtered_community; tw=tw, deterministic=deterministic, lag=lag, α=α);
        if(feasib_criterion & ss_criterion)
            conds_met = true
        else
            conds_met = false
        end    
    end
    
    return conds_met
    
end


function generateNewlyAssembledCommunity(init_param_struct,sol;set_intrasps_effects=-1)
    "Create augmented community, which carries a successful invader, which is feasible, and the collective dynamics of which
     enter into a steady state
    "
    let augmented_param_struct, sol_invaded_community, ss_abundances;
        cnac = false;
        while(cnac==false)
            augmented_param_struct, sol_invaded_community, ss_abundances = addNewSps2ResidentCommunity(init_param_struct,
                                                                                                      set_intrasps_effects=set_intrasps_effects);
            cnac = checkingNewlyAssembledCommunity(augmented_param_struct,ss_abundances);
        end
        
        return augmented_param_struct, sol_invaded_community, ss_abundances
    
    end

end


function assess_topol_params_compet_net(params_struct; int_str_thr=0.75)
    "Use function to assess some informative network metrics"
    
    A = copy(params_struct["connectMatrix"]);

    #Assess skewness of competitive interaction distributions in the network
    fA = collect(Iterators.flatten(A));
    SKN = skewness(fA[fA.!=-1]);
    
    #Difference in entropy compared to a uniform distribution: A large deviation from the uniform distribution 
    #indicates that one or more interactions dominate the network
    DE2U = diff_entropy_uniform(A);
   
    #Let's apply a threshold to the distribution of interaction strengths in the network.
    #We assume here that e.g. only the entries in the matrix |α{ij}| >= Q3 are the main drivers of community dynamics. See thresholded_matrix()
    _,thresholdedA = thresholded_matrix(A, int_str_thr=int_str_thr);
    
    #Assess nestedness: spectral radius (the absolute value of the largest real part of the eigenvalues of the adjacency matrix) 
    #of any unipartite network whose interactions are positive or null.
    #Here the spectral radius is divided by the square root of (2L(S-1))/S 
    #(Phillips, J.D., 2011. The structure of ecological state transitions: Amplification, synchronization, and constraints in responses to environmental change.)
    NEST = EcologicalNetworks.ρ_phillips(UnipartiteNetwork(thresholdedA),ρ(UnipartiteNetwork(thresholdedA)));  

    #Computes the heterogeneity for an unipartite network, a topological characteristic which quantifies 
    #the difference in in- and out-going degrees between species.
    #It is computed as σin * σout / s_mean. A value of 0 indicates that all species have the same (weighted) in- and outdegrees.
    HET = heterogeneity(UnipartiteNetwork(thresholdedA));    
    
    #Check connectance after thresholding the distribution of interaction strengths. Check functions effect_pairwise_sps_int and effect_connect 
    TRHCNT = effect_connect(A,int_str_thr);
    # TRHCNT = EcologicalNetworks.connectance(UnipartiteNetwork(thresholdedA))
    
    return SKN,DE2U,NEST,HET,TRHCNT
end


function vis_sps_interact_net(cnm; int_str_thr=0.85,layout=:circular)
    "Use function to visualize ecological networks using various types of layouts"

    #Grab original diagonal values
    intrinsic_sps_growth = copy(diag(cnm));

    #Set self-loops to 0, to avoid these being displayed
    cnm[diagind(cnm)] .= 0;

    #Let's apply a threshold to the distribution of interaction strengths in the network.
    #We assume that only the entries in the matrix α{ij} >= Q3 are the main drivers of community dynamics
    unique_coeffs = (sort(unique(cnm)));
    cutoff = quantile(abs.(unique_coeffs),int_str_thr)
    thresholdedA = (abs.(cnm) .>= cutoff);
    new_cnm = sign.(cnm).*thresholdedA;

    #Create graph object out of the adjacency matrix
    g = LightGraphs.DiGraph(new_cnm)
    #Let's change edge color appropriately: create an empty array to store edge color
    edge_colors_mat = Array{Symbol,2}(undef,size(new_cnm)...);
    for e in LightGraphs.edges(g)
        if(new_cnm[e.src,e.dst]<0)
            edge_colors_mat[e.src,e.dst]=:red
        else
            edge_colors_mat[e.src,e.dst]=:blue
        end
    end

    #Plot graph; choose among the following layouts: method=:circular or method=:arcdiagram or method=:chorddiagram
    p = GraphRecipes.graphplot(g, 
                               nodeshape=:hexagon,
                               names=["Sps$i" for i in 1:size(new_cnm,1)],
                               nodesize=0.075,
                               nodecolor = range(colorant"green", stop=colorant"yellow", length=size(new_cnm,1)),
                               curvature_scalar=0.05, 
                               shorten=0.04, 
                               arrow=Plots.arrow(:closed, :tail, 0.01, 0.01),
                               edgecolor=edge_colors_mat,
                               method=layout
                               )
    #Restore original instrinsic growth rates
    cnm[diagind(cnm)] .= intrinsic_sps_growth

    plot(p,size=(900,900))
    
end


function thresholded_matrix(connect_mat; int_str_thr=0.75)
    "Use function to threshold a matrix by setting the entries to 0 if abs value of each entry is < than a threshold value
     int_str_thr, which we assume to be specified by the quantile of the distribution of absolute interaction strengths in the
     matrix. The function returns the signed and unsigned thersholded matrices for the input matrix connect_mat"
     
     cnm = copy(connect_mat);
     #Set self-loops to 0, to avoid these being displayed
     cnm[diagind(cnm)] .= 0;
 
     #Let's apply a threshold to the distribution of interaction strengths in the network.
     #We assume here that e.g. only the entries in the matrix |α{ij}| >= Q3 are the main drivers of community dynamics
     unique_coeffs = (sort(unique(cnm)));
     cutoff = quantile(abs.(unique_coeffs),int_str_thr)
     thresholdedA = (abs.(cnm) .>= cutoff);
     new_cnm = sign.(cnm).*thresholdedA;    
     
     return new_cnm, convert(Array,thresholdedA)
 end

function effect_pairwise_sps_int(connect_mat; int_str_thr=0.75)
    "Use function to calculate total number of effective pairwise sps interactions in the network, 
     the interaction strengths of which are |aij| >= int_str_thr. Note that, if int_str_thr=0.0
     the total number of effective pairwise species interaction would be the maximal possible" 
  
     new_cnm,_ = thresholded_matrix(connect_mat, int_str_thr=int_str_thr)
 
     #Create graph object out of the adjacency matrix
     g = LightGraphs.DiGraph(new_cnm)
     n=0;
     for e in LightGraphs.edges(g)
         if((new_cnm[e.src,e.dst]!=0) && new_cnm[e.dst,e.src]!=0)
             n+=1;
             #println([new_cnm[e.src,e.dst],new_cnm[e.dst,e.src]])
         end
     end
     return n
 end

function effect_connect(cnm,thr)
    "Function that calculates effective network connectance/connectivity or density of interactions"
    effect_pairwise_sps_int(cnm; int_str_thr=thr)/prod(size(cnm))
end

function runAdaptRadiationCommunAssembly(init_size,target_size;
                                         sample_range_A = (-1.,-1e-3),
                                         sample_range_grs = (1e-1,1e0),
                                         sample_range_ics = (1e-1,1e0),
                                         edge_connect_probab = rand(Uniform(0.5,1)),
                                         integ_steps = 2000, ub_density = 1e1, 
                                         lb_density = 1e-3, tw=1:2000, 
                                         deterministic=:constant, 
                                         lag=0, α=0.05,
                                         prob_scale=0.5,
                                         set_intrasps_effects=-1,
                                         perturb_intrinsic_grs=false,
                                         iter_limit=100,
                                         A_symmetric = true,
                                         dyn_var_thr = 1e-4,
                                         repl = 1,
                                         output_folder = "../output/compet_commun_assembly_full_record/")
    
    "Function to run the adaptive radiation algorithm for assemblying a (relatively small) community 
     with a given number of strictly competitively interacting species"
    
    #Check if output folder exists!
    if(!isdir(output_folder))
        mkdir(output_folder)
    end
    
    let init_param_struct,CS;
        
        #Dict to grab all param structures
        nested_dict = Dict();
        
        #Generate seed community of a given size and connectivity
        init_param_struct,sol = rndGenerationInitSpsNetwork(init_size, 
                                                            sample_range_A=sample_range_A,
                                                            sample_range_grs=sample_range_grs,
                                                            sample_range_ics=sample_range_ics,
                                                            edge_connect_probab=edge_connect_probab, 
                                                            integ_steps=integ_steps, 
                                                            ub_density=ub_density, 
                                                            lb_density=lb_density, tw=tw, 
                                                            deterministic=deterministic, 
                                                            lag=lag, α=α, A_symmetric=A_symmetric);
        
        
        #Grab initially generated community network
        nested_dict[0] = init_param_struct;
        
#         println("Ready to start community assembly process strictly competitive interaction network...\n");
        
        #Run main loop where community is assembled progressively by adding new species duplicated from pre-existing ones in the
        #resident community
        
        #MCS: mean interspecies competition strength
        MCS = Any[];
        A = copy(init_param_struct["connectMatrix"])
        A[diagind(A)] .= 0.0;
        push!(MCS,(sum(A))/(init_size*(init_size-1)));
        
        #Reset diagonal elements to intraspecific competition for resources 
        A[diagind(A)] .= -1.0;
        init_param_struct["connectMatrix"]=A;

        #CS: Comunity size
        CS = Any[];
        push!(CS,init_size);#Add initial size
  
        #Total biomass, T, taken as overall sum of steady state species abundances in the community.
        #See: Generic assembly patterns in complex ecological communities
        T = Any[];
        ss_abundances_init_community = sol(init_param_struct["endTimePoint"]);
        push!(T,sum(ss_abundances_init_community));
        
        #Temporal variance of species abundance, referred to as ecological stability in: 
        #Generic assembly patterns in complex ecological communities
        #Here we set a threshold for temporal variance of species abundance trajectories.
        #This is to favor communities showing individual trajectories with some degree of 
        #temporal variation, which may be driven by interactions with other species in the network
        V = Any[];
        full_ts_df = DataFrame(convert(Array,DataFrame([sol(i) for i in 1:init_param_struct["endTimePoint"]]))');
        variance_per_ts = convert(Array,mapcols(col -> var(col), full_ts_df))
        eco_stab = mean(variance_per_ts);
        push!(V,eco_stab)        
        
        #Assessing other network metrics: 
        # - SKN: Skewness of competitive interaction distributions in the network
        # - DE2U: Difference in entropy compared to a uniform distribution
        # - NEST: nestedness, the spectral radius of the connectivity matrix
        # - HET: topological characteristic which quantifies the difference in in- and out-going degrees between species
        SKN = Any[];
        DE2U = Any[];
        NEST = Any[];
        HET = Any[];
        TRHCNT = Any[]; 
        skn,de2u,nest,het,trhc = assess_topol_params_compet_net(init_param_struct);
        push!(SKN,skn);
        push!(DE2U,de2u);
        push!(NEST,nest);
        push!(HET,het);
        push!(TRHCNT,trhc);

        #Grab number of extinction events here
        NUMEXT = Any[];
        num_sps = init_size;
        n=0;
        while num_sps < target_size
            n = n+1;
            current_size = length(init_param_struct["growthRates"]);
            augmented_param_struct, sol_invaded_community, ss_abundances = addNewSps2ResidentCommunity(init_param_struct,
                                                                                                       set_intrasps_effects=set_intrasps_effects,
                                                                                                       perturb_intrinsic_grs=perturb_intrinsic_grs);    
            
            augmented_param_struct = pruningCommunity4ExtinctSps(augmented_param_struct,ss_abundances);
            sol_invaded_community = checkSol4DynSyst(augmented_param_struct,ub_density);
            ss_abundances = sol_invaded_community(augmented_param_struct["endTimePoint"]);
            cnac = checkingNewlyAssembledCommunity(augmented_param_struct,ss_abundances);
            num_sps = length(augmented_param_struct["growthRates"]);

            if(!cnac || num_sps < init_size)
                init_param_struct = init_param_struct;
            else
                #Accept if current community size is >= than previous engineered community &&
                ##The minimal temporal variance i abundance is above a predefined threshold
                if((num_sps >= CS[end]) && (minimum(variance_per_ts) >= dyn_var_thr))
#                     println("Iter $n, current community size = $num_sps"); 
                    init_param_struct =  augmented_param_struct;
                    
                #Otherwise, accept a community with smaller size (i.e. with extinct sps) with a given probab 
                #Here we apply the Metropolis scheme as: rnd < exp(-d/k), with d denoting the difference in
                #community size between current and previously registered community
                else
                    d = num_sps - CS[end];
                    #Store No. extinction events driven by current invader 
                    push!(NUMEXT,abs(d));
                    if(rand() < exp(d/prob_scale))
                    println("Iter $n, current community size = $num_sps"); 
                        init_param_struct =  augmented_param_struct;
                    end
                end
            end
            
            #Grab parameters for newly created community network
            nested_dict[n] = init_param_struct;
            
            #MCS: mean interspecies competition strength
            A = copy(init_param_struct["connectMatrix"])
            A[diagind(A)] .= 0.0;
            push!(MCS,(sum(A))/(num_sps*(num_sps-1)));
            
            #Reset diagonal elements to intraspecific competition for resources 
            A[diagind(A)] .= -1.0;
            init_param_struct["connectMatrix"]=A;            

            #Check if current community size is larger than any other community assembled up until now
            #If true, then dump into files all stats/descriptors registered so far 
            CSCond = num_sps>maximum(CS); 
            #Record community size during the assembly process
            push!(CS,num_sps);
            
            #Assess total biomass, T, taken as overall sum of steady state species abundances in the community.
            sol_current_community = checkSol4DynSyst(init_param_struct,ub_density);
            ss_abundances_current_community = sol_current_community(init_param_struct["endTimePoint"]);
            push!(T,sum(ss_abundances_current_community));
            
            #Assess ecological stability. See: Generic assembly patterns in complex ecological communities
            full_ts_df = DataFrame(convert(Array,DataFrame([sol_current_community(i) for i in 1:init_param_struct["endTimePoint"]]))');
            variance_per_ts = convert(Array,mapcols(col -> var(col), full_ts_df))
            eco_stab = mean(variance_per_ts);
            push!(V,eco_stab);
            
            skn,de2u,nest,het,trhc = assess_topol_params_compet_net(init_param_struct);
            push!(SKN,skn);
            push!(DE2U,de2u);
            push!(NEST,nest);
            push!(HET,het);
            push!(TRHCNT,trhc);

            #Set max iterations
            if(CSCond)  
                #Saving data
                JLD.save(output_folder*"ParamStructCCAssemblyStandGralLV_Rep$repl.jld", "data", nested_dict);
                
                df = DataFrame(commun_size = CS, 
                               ecol_stabil = V, 
                               total_biom = T, 
                               mean_intsps_comp_str = MCS,
                               skn_int_dist = SKN,
                               entropy_diff = DE2U,
                               nestedness = NEST,
                               heter_in_out_degrees = HET,
                               thr_connect = TRHCNT
                               );
                df[!, :commun_size] = convert.(Int, df[!, :commun_size]);
                df[!, :ecol_stabil] = convert.(Float64, df[!, :ecol_stabil]);
                df[!, :total_biom] = convert.(Float64, df[!, :total_biom]);
                df[!, :mean_intsps_comp_str] = convert.(Float64, df[!, :mean_intsps_comp_str]);
                df[!, :skn_int_dist] = convert.(Float64, df[!, :skn_int_dist]);
                df[!, :entropy_diff] = convert.(Float64, df[!, :entropy_diff]);
                df[!, :nestedness] = convert.(Float64, df[!, :nestedness]);
                df[!, :heter_in_out_degrees] = convert.(Float64, df[!, :heter_in_out_degrees]);
                df[!, :thr_connect] = convert.(Float64, df[!, :thr_connect]);
                save(output_folder*"StatsCCAssemblyStandGralLV_Rep$repl.csv", df)
                CSV.write(output_folder*"NumExtinctionEvents_Rep$repl.csv",DataFrame(num_ext_events = NUMEXT))
            end
            
            #Break out loop if max num iters is attained
            if(n==iter_limit)
                break;
            end
        end

    end
end


function runAdaptRadiationCommunAssembly2(init_size,target_size;
                                         sample_range_A = (-1.,-1e-3),
                                         sample_range_grs = (1e-1,1e0),
                                         sample_range_ics = (1e-1,1e0),
                                         edge_connect_probab = rand(Uniform(0.5,1)),
                                         integ_steps = 2000, ub_density = 1e1, 
                                         lb_density = 1e-3, tw=1:2000, 
                                         deterministic=:constant, 
                                         lag=0, α=0.05,
                                         prob_scale=0.5,
                                         set_intrasps_effects=-1,
                                         perturb_intrinsic_grs=false,
                                         iter_limit=100,
                                         A_symmetric = true,
                                         add_mutual_inters = true,
                                         add_mutualism_perc = 0.2, 
                                         mutual_strength_range = [1e-3,1e0],
                                         dyn_var_thr = 1e-4,
                                         repl = 1,
                                         output_folder = "../output/compet_mutual_commun_assembly_full_record/")
    
    "Function to run the adaptive radiation algorithm for assemblying a (relatively small) community with a 
    given number of competitively interacting species plus a few mutualistic interacting species."
    
    #Check if output folder exists!
    if(!isdir(output_folder))
        mkdir(output_folder)
    end    
    
    let init_param_struct,CS;
        
        #Dict to grab all param structures
        nested_dict = Dict();        
        
        #Generate seed community of a given size and connectivity
        init_param_struct,sol = rndGenerationInitSpsNetwork(init_size, 
                                                            sample_range_A=sample_range_A,
                                                            sample_range_grs=sample_range_grs,
                                                            sample_range_ics=sample_range_ics,
                                                            edge_connect_probab=edge_connect_probab, 
                                                            integ_steps=integ_steps, 
                                                            ub_density=ub_density, 
                                                            lb_density=lb_density, tw=tw, 
                                                            deterministic=deterministic, 
                                                            lag=lag, α=α, A_symmetric=A_symmetric,
                                                            add_mutual_inters = add_mutual_inters,
                                                            add_mutualism_perc = add_mutualism_perc, 
                                                            mutual_strength_range = mutual_strength_range);
        
        #Grab initially generated community network
        nested_dict[0] = init_param_struct;        
        
#         println("Ready to start community assembly process for prominently competitive + moderately mutualistic interaction network...\n");
        
        #Run main loop where community is assembled progressively by adding new species duplicated from pre-existing ones in the
        #resident community
        
        #MCS: mean interspecies competition strength
        MCS = Any[];
        A = copy(init_param_struct["connectMatrix"])
        A[diagind(A)] .= 0.0;
        push!(MCS,(sum(A[A .< 0]))/(init_size*(init_size-1)));    

        #MMS: mean interspecies mutualistic strength
        MMS = Any[];
        push!(MMS,(sum(A[A .> 0]))/(init_size*(init_size-1)));  
        
        #CS: Comunity size
        CS = Any[];
        push!(CS,init_size);#Add initial size
  
        #Reset diagonal elements to intraspecific competition for resources 
        A[diagind(A)] .= -1.0;
        init_param_struct["connectMatrix"]=A;    

        #Total biomass, T, taken as overall sum of steady state species abundances in the community.
        #See: Generic assembly patterns in complex ecological communities
        T = Any[];
        ss_abundances_init_community = sol(init_param_struct["endTimePoint"]);
        push!(T,sum(ss_abundances_init_community));
        
        #Temporal variance of species abundance, referred to as ecological stability in: 
        #Generic assembly patterns in complex ecological communities
        #Here we set a threshold for temporal variance of species abundance trajectories.
        #This is to favor communities showing individual trajectories with some degree of 
        #temporal variation, which may be driven by interactions with other species in the network
        V = Any[];
        full_ts_df = DataFrame(convert(Array,DataFrame([sol(i) for i in 1:init_param_struct["endTimePoint"]]))');
        variance_per_ts = convert(Array,mapcols(col -> var(col), full_ts_df))
        eco_stab = mean(variance_per_ts);
        push!(V,eco_stab)        
        
        #Assessing other network metrics: 
        # - SKN: Skewness of competitive interaction distributions in the network
        # - DE2U: Difference in entropy compared to a uniform distribution
        # - NEST: nestedness, the spectral radius of the connectivity matrix
        # - HET: topological characteristic which quantifies the difference in in- and out-going degrees between species
        #SKN = Any[];
        DE2U = Any[];
        NEST = Any[];
        HET = Any[];
        TRHCNT = Any[]; 
        _,de2u,nest,het,trhc = assess_topol_params_compet_net(init_param_struct);
        #push!(SKN,skn);
        push!(DE2U,de2u);
        push!(NEST,nest);
        push!(HET,het);
        push!(TRHCNT,trhc);
        
        #Grab number of extinction events here
        NUMEXT = Any[];
        num_sps = init_size;
        n=0;
        while num_sps < target_size
            n = n+1;
            current_size = length(init_param_struct["growthRates"]);
            augmented_param_struct, sol_invaded_community, ss_abundances = addNewSps2ResidentCommunity(init_param_struct,
                                                                                                       set_intrasps_effects=set_intrasps_effects,
                                                                                                       perturb_intrinsic_grs=perturb_intrinsic_grs);    
            
            augmented_param_struct = pruningCommunity4ExtinctSps(augmented_param_struct,ss_abundances);
            sol_invaded_community = checkSol4DynSyst(augmented_param_struct,ub_density);
            ss_abundances = sol_invaded_community(augmented_param_struct["endTimePoint"]);
            cnac = checkingNewlyAssembledCommunity(augmented_param_struct,ss_abundances);
            num_sps = length(augmented_param_struct["growthRates"]);

            if(!cnac || num_sps < init_size)
                init_param_struct = init_param_struct;
            else
                #Accept if current community size is >= than previous engineered community &&
                ##The minimal temporal variance i abundance is above a predefined threshold
                if((num_sps >= CS[end]) && (minimum(variance_per_ts) >= dyn_var_thr))
                    #println("Iter $n, current community size = $num_sps"); 
                    init_param_struct =  augmented_param_struct;
                    
                #Otherwise, accept a community with smaller size (i.e. with extinct sps) with a given probab 
                #Here we apply the Metropolis scheme as: rnd < exp(-d/k), with d denoting the difference in
                #community size between current and previously registered community
                else
                    d = num_sps - CS[end];
                    #Store No. extinction events driven by current invader 
                    push!(NUMEXT,abs(d));
                    if(rand() < exp(d/prob_scale))
                        #println("Iter $n, current community size = $num_sps"); 
                        init_param_struct =  augmented_param_struct;
                    end
                end
            end
            
            #Grab parameters for newly created community network
            nested_dict[n] = init_param_struct;            
            
            #MCS: mean interspecies competition strength
            A = copy(init_param_struct["connectMatrix"])
            A[diagind(A)] .= 0.0;
            push!(MCS,(sum(A[A .< 0]))/(init_size*(init_size-1)));
            push!(MMS,(sum(A[A .> 0]))/(init_size*(init_size-1)));
            
            #Reset diagonal elements to intraspecific competition for resources 
            A[diagind(A)] .= -1.0;
            init_param_struct["connectMatrix"]=A;

            #Check if current community size is larger than any other community assembled up until now
            #If true, then dump into files all stats/descriptors registered so far 
            CSCond = num_sps>maximum(CS); 
            #Record community size during the assembly process
            push!(CS,num_sps);
            
            #Assess total biomass, T, taken as overall sum of steady state species abundances in the community.
            sol_current_community = checkSol4DynSyst(init_param_struct,ub_density);
            ss_abundances_current_community = sol_current_community(init_param_struct["endTimePoint"]);
            push!(T,sum(ss_abundances_current_community));
            
            #Assess ecological stability. See: Generic assembly patterns in complex ecological communities
            full_ts_df = DataFrame(convert(Array,DataFrame([sol_current_community(i) for i in 1:init_param_struct["endTimePoint"]]))');
            variance_per_ts = convert(Array,mapcols(col -> var(col), full_ts_df))
            eco_stab = mean(variance_per_ts);
            push!(V,eco_stab);
            
            _,de2u,nest,het,trhc = assess_topol_params_compet_net(init_param_struct);
            #push!(SKN,skn);
            push!(DE2U,de2u);
            push!(NEST,nest);
            push!(HET,het);
            push!(TRHCNT,trhc);

            #Set max iterations
            if(CSCond)  
                #Saving data
                JLD.save(output_folder*"ParamStructCCAssemblyStandGralLV_Rep$repl.jld", "data", nested_dict);
                
                df = DataFrame(commun_size = CS, 
                               ecol_stabil = V, 
                               total_biom = T, 
                               mean_intsps_comp_str = MCS,
                               mean_intsps_mutual_str = MMS,
                               #skn_int_dist = SKN,
                               entropy_diff = DE2U,
                               nestedness = NEST,
                               heter_in_out_degrees = HET,
                               thr_connect = TRHCNT,
                               );
                df[!, :commun_size] = convert.(Int, df[!, :commun_size]);
                df[!, :ecol_stabil] = convert.(Float64, df[!, :ecol_stabil]);
                df[!, :total_biom] = convert.(Float64, df[!, :total_biom]);
                df[!, :mean_intsps_comp_str] = convert.(Float64, df[!, :mean_intsps_comp_str]);
                df[!, :mean_intsps_mutual_str] = convert.(Float64, df[!, :mean_intsps_mutual_str]);
                #df[!, :skn_int_dist] = convert.(Float64, df[!, :skn_int_dist]);
                df[!, :entropy_diff] = convert.(Float64, df[!, :entropy_diff]);
                df[!, :nestedness] = convert.(Float64, df[!, :nestedness]);
                df[!, :heter_in_out_degrees] = convert.(Float64, df[!, :heter_in_out_degrees]);
                df[!, :thr_connect] = convert.(Float64, df[!, :thr_connect]);
                save(output_folder*"StatsCCAssemblyStandGralLV_Rep$repl.csv", df)
                CSV.write(output_folder*"NumExtinctionEvents_Rep$repl.csv",DataFrame(num_ext_events = NUMEXT))
                
            end
            
            #Break out loop if max num iters is attained
            if(n==iter_limit)
                break;
            end
        end

    end
end
