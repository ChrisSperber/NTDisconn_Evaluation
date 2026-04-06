# Modification of NTDisconn
This repository is a fork of (https://github.com/phjkoch/NTDisconn), commit 9411865.
It contains customizations for the application of the NTDisconn Tool on a scientific study by the University Hopsital of Bern.
The original authors are credited below and the original license applies.

## Modifications
Following modifications were introduced
- new --nt_hotspot_t flag in the Create_NTDisconn.py main pipeline
- new --subtract_disc_fraction flag in the Create_NTDisconn.py main pipeline
- refactoring/cleanup of the Create_NTDisconn.py main function
- addition of various diagnostic scripts used during the evaluation of the NTDisconn functionality

### --nt_hotspot_t flag
optional hotspot-emphasizing transform of per-streamline NT weights

If provided, the t value is used as the robust-z threshold in:
```
z = (x - median(x)) / (1.4826 * MAD(x))
w = sigmoid(alpha * (z - t))
```

Intuition:
- larger t -> only strong "hotspot" weights remain influential
- smaller t -> more weights contribute (closer to original behavior)

This transform is only supported in --NTmaps Percent mode (proportion-style output).
If --NTmaps Z is used together with this option, the script will fail.

### --subtract_disc_fraction flag
subtract global disconnection burden (F) from each NT percent score

In Percent mode, the base score is:#
```
P_k = sum(disconnected * w_k) / sum(w_k)
```
where disconnected is the 0/1 streamline disconnection mask.

The global disconnection fraction is:
```
F = (# disconnected streamlines) / (total # streamlines)
```
If enabled, the script outputs:
```
P_k - F
```

This removes the shared component driven purely by "how many streamlines are disconnected".
Values can become negative.

## review-clean branch
An additional branch review-clean has been added that, compared to the original repo, only contains the main functional changes introduced by the
nt_hotspot_t and subtract_disc_fraction flags

```bash
git checkout review-clean
```

## Original Authors and License Information

Create Neurotransmitter Network Damage  
NTDisconn ©️ 2025 by Philipp J. Koch is licensed under CC BY-NC-SA 4.0  
[https://creativecommons.org/licenses/by-nc-sa/4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)  

This work is based on the following and distributed under the CC BY-NC-SA 4.0 License:  
Please cite the following when using this code  
- Neurotransmitter informed Network Damage for stroke outcome:   
[Koch and Frey et al. 2025](https://pubmed.ncbi.nlm.nih.gov/40392973/) (https://pubmed.ncbi.nlm.nih.gov/40392973/) 
- Population based Tractogram:    
[Xiao et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10474320/) (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10474320/)      
[This data is available here](https://osf.io/p7syt/) (https://osf.io/p7syt/)    
- Neurotransmitter Density maps:    
[Hansen et al. 2022](https://pubmed.ncbi.nlm.nih.gov/36303070/) (https://pubmed.ncbi.nlm.nih.gov/36303070/)    
[Git Repository](https://github.com/netneurolab/hansen_receptors/tree/main) (https://github.com/netneurolab/hansen_receptors/tree/main)    
