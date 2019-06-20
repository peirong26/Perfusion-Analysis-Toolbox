% DSC_mri_toolbox demo

%datapath = '/Users/peirong/Documents/MATLAB/dsc-mri-toolbox/demo-data';
datapath = '/Users/peirong/Downloads/ISLES/ISLES2017_training/training_1/VSD.Brain.XX.O.MR_4DPWI.127015';
% ------ Load the dataset to be analyzed ---------------------------------
%DSC_info   = niftiinfo(fullfile(datapath,'GRE_DSC.nii.gz'));
DSC_info   = niftiinfo(fullfile(datapath,'VSD.Brain.XX.O.MR_4DPWI.127015.nii'));
DSC_volume = niftiread(DSC_info);
R = length(DSC_volume(:,1,1,1,1));
V = length(DSC_volume(1,:,1,1,1));
S = length(DSC_volume(1,1,:,1,1));
T = length(DSC_volume(1,1,1,1,:));
DSC_volume = reshape(DSC_volume, [R, V, S, T]);

% ------ Set minimum acquistion parameters -------------------------------
TE = 0.025; % 25ms
TR = 1.55;  % 1.55s

% ------ Perform quantification ------------------------------------------ 
% Input   DSC_volume (4D matrix with raw GRE-DSC acquisition)
%         TE         (Echo time)
%         TR         (Repetition time)
% Output  cbv        (3D matrix with standard rCBV values)
%         cbf        (struct with 3D matrices of rCBF values for each method selected)
%         mtt        (struct with 3D matrices of MTT values for each method selected)
%         cbv_lc     (3D matrix with leackage corrected rCBV values)
%         ttp        (3D matrix with leackage corrected Time to Peak values)
%         mask       (3D matrix with computed mask)
%         aif        (struct with AIF extracted with clustering algorithm)
%         conc       (4D matrix with pseudo-concentration values)
%         s0         (3D matrix with S0 estimates from pre-contrast images)

[cbv,cbf,mtt,cbv_lc,ttp,mask,aif,conc,s0]=DSC_mri_core(DSC_volume,TE,TR);
cbf_svd = cbf.svd.map;
cbf_csvd = cbf.csvd.map;
cbf_osvd = cbf.osvd.map;
mtt_svd = mtt.svd;
mtt_csvd = mtt.csvd;
mtt_osvd = mtt.osvd;
% ------  Save Results --------------------------------------------------- 
niftiwrite(cbv,fullfile(datapath,'CBV.nii'));
niftiwrite(cbv_lc,fullfile(datapath,'CBV_LC.nii'));
niftiwrite(cbf_svd,fullfile(datapath,'CBF_SVD.nii'));
niftiwrite(cbf_csvd,fullfile(datapath,'CBF_CSVD.nii'));
niftiwrite(cbf_osvd,fullfile(datapath,'CBF_OSVD.nii'));
niftiwrite(mtt_svd,fullfile(datapath,'MTT_SVD.nii'));
niftiwrite(mtt_csvd,fullfile(datapath,'MTT_CSVD.nii'));
niftiwrite(mtt_osvd,fullfile(datapath,'MTT_OSVD.nii'));
niftiwrite(ttp,fullfile(datapath,'TTP.nii'));

% ------  View Results --------------------------------------------------- 
DSC_mri_show_results(cbv_lc,cbf,mtt,ttp,mask,aif,conc,s0);
