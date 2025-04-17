# NTDisconn
Create Neurotransmitter Network Damage  
NTDisconn ©️ 2025 by Philipp J. Koch is licensed under CC BY-NC-SA 4.0  
[https://creativecommons.org/licenses/by-nc-sa/4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)  

This work is based on the following and distributed under the CC BY-NC-SA 4.0 License:  
Please cite the following when using this code  
- Population based Tractogram:    
[Xiao et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10474320/) (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10474320/)      
[This data is available here](https://osf.io/p7syt/) (https://osf.io/p7syt/)    
- Neurotransmitter Density maps:    
[Hansen et al. 2022](https://pubmed.ncbi.nlm.nih.gov/36303070/) (https://pubmed.ncbi.nlm.nih.gov/36303070/)    
[Git Repository](https://github.com/netneurolab/hansen_receptors/tree/main) (https://github.com/netneurolab/hansen_receptors/tree/main)    



1. Required repositories  
&nbsp;&nbsp;&nbsp;&nbsp;nibabel  
&nbsp;&nbsp;&nbsp;&nbsp;scipy  
&nbsp;&nbsp;&nbsp;&nbsp;dipy  
&nbsp;&nbsp;&nbsp;&nbsp;antspyx  
&nbsp;&nbsp;&nbsp;&nbsp;pandas  
&nbsp;&nbsp;&nbsp;&nbsp;alternative: activate the given environment (s. 3.)

3. Clone the repository
```bash
git clone https://github.com/phjkoch/NTDisconn.git
cd NTDisconn
```

3. Activate Environment
```bash
conda env create -f environment.yml
conda activate ntdisconn
```

4. Usage

```bash
python Create_NTDisconn.py --help
```
&nbsp;&nbsp;&nbsp;&nbsp;Usage:  
&nbsp;&nbsp;&nbsp;&nbsp;Create_NTDisconn.py ID in_lesion output_dir  

&nbsp;&nbsp;&nbsp;&nbsp;positional arguments:  
&nbsp;&nbsp;&nbsp;&nbsp;ID:                    Subject ID  
&nbsp;&nbsp;&nbsp;&nbsp;in_lesion:             Input individual Lesionmask in MNI152 (1mm iso)  
&nbsp;&nbsp;&nbsp;&nbsp;output_dir:            Specify output directory

&nbsp;&nbsp;&nbsp;&nbsp;optional arguments:  
&nbsp;&nbsp;&nbsp;-h, --help            show this help message and exit  
&nbsp;&nbsp;&nbsp;--discStreamlines DISCSTREAMLINES
                    Create disconnected streamline output? [y|n] (default: y)  
&nbsp;&nbsp;&nbsp;--NTmaps MAPS
                    Choose which NT maps to use (Z-values vs. Percentage)? [Z|Percent] (default: Percent)                    

5. Output  
&nbsp;&nbsp;&nbsp;In the output_dir a directory named after the ID is created containing  
&nbsp;&nbsp;&nbsp;1. A csv file with the estimated Neurotransmitter network damage of the individual lesion map for all the Neurotransmitter receptors and transporters from Hansen et al. 2022  
&nbsp;&nbsp;&nbsp;2. A txt file with 2 millionen entries indicating which streamlines of the HCP-aging tractogram is passing through the individual lesion mask [1] and which are sparsed [0] (optional)


6. Test_MNI_lesion.nii.gz
This is a test lesion when used correctly like:
```bash
python Create_NTDisconn.py Test Test_MNI_lesion.nii.gz output_test
```

Creates the individual Neurotransmitter network damage:


