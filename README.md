# DAG-Net: Double Attentive Graph Neural Network for Trajectory Forecasting
This repository contains the PyTorch code for [ICPR 2020](https://www.micc.unifi.it/icpr2020/) paper:

**<a href="https://arxiv.org/abs/2005.12661v2">DAG-Net: Double Attentive Graph Neural Network for Trajectory Forecasting</a>**  
*<a href="https://github.com/alexmonti19">Alessio Monti</a>,
<a href="https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=110">Alessia Bertugli</a>,
<a href="https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=38">Simone Calderara</a>,
<a href="https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=1">Rita Cucchiara</a>*  

## Model architecture
The model is composed by three main components: the generative model (VRNN) and two graph neural networks.

![dagnet - overview](icpr.png)
 
The first graph network operates on agents' goals, expressed as specific areas of the environment where agents 
will land in the future. The second second graph network operates instead on single agents' hidden states, 
which contain past motion behavioural information. The employing of two separate graph neural networks allows to consider 
and share both past and future information while generating agents' future movements.


## Prerequisites

* Python >= 3.8
* PyTorch >= 1.5
* CUDA 10.0

### Installation

* Clone this repo:
```
git clone https://github.com/alexmonti19/dagnet.git
cd dagnet
```

* Create a new virtual environment using Conda or virtualenv. 
```
conda create --name <envname>
```
* Activate the environment and install the requirements:
```
conda activate <envname>
pip install -r requirements.txt
```


## Datasets
Our proposal is general enough to be applied in different scenarios: the model achieves state-of-the-art results in both
urban environments (*Stanford Drone Dataset*) and sports applications (*STATS SportVU NBA Dataset*).   

For complete information on where to download and how to preprocess the datasets see the relative 
[datasets/README.md](./datasets/README.md).

## Architectures
The repo contains both the final model and the two ablation architectures cited in the paper.
- *VRNN*: the baseline generative architecture
- *A-VRNN*: enhanced version of the baseline with a single graph (on agents' hidden states)
- *DAG-Net*: the complete architecture with two graphs

For more information on how to train the models see the relative [models/README.md](./models/README.md).

## Pre-trained models
We also ship the checkpoints to reproduce the results presented in the paper. 

<table>
    <tbody>
        <tr>
            <th>
            <th align="center" colspan=2 style="text-align:center">Offense BSK</th>
            <th align="center" colspan=2 style="text-align:center">Defense BSK</th>
            <th align="center" colspan=2 style="text-align:center">Stanford Drone</th>
        </tr>
        <tr>
            <td align="center"></td>
            <td align="center" style="text-align:center; font-weight:bold">ADE</td>
            <td align="center" style="text-align:center; font-weight:bold">FDE</td>
            <td align="center" style="text-align:center; font-weight:bold">ADE</td>
            <td align="center" style="text-align:center; font-weight:bold">FDE</td>
            <td align="center" style="text-align:center; font-weight:bold">ADE</td>
            <td align="center" style="text-align:center; font-weight:bold">FDE</td>
        </tr>
        <tr>
            <td>VRNN</td>
            <td style="text-align:center">9,41</td>
            <td style="text-align:center">15,56</td>
            <td style="text-align:center">7,16</td>
            <td style="text-align:center">10,50</td>
            <td style="text-align:center">0,58</td>
            <td style="text-align:center">1,17</td>
        </tr>
        <tr>
            <td>A-VRNN</td>
            <td style="text-align:center">9,48</td>
            <td style="text-align:center">15,52</td>
            <td style="text-align:center">7,05</td>
            <td style="text-align:center">10,34</td>
            <td style="text-align:center">0,56</td>
            <td style="text-align:center">1,14</td>
        </tr>
        <tr>
            <td>DAG-Net</td>
            <td style="text-align:center"><em>8,98</em></td>
            <td style="text-align:center"><em>14,08</em></td>
            <td style="text-align:center"><em>6,87</em></td>
            <td style="text-align:center"><em>9,76</em></td>
            <td style="text-align:center"><em>0,53</em></td>
            <td style="text-align:center"><em>1,04</em></td>
        </tr>
    </tbody>
</table>

For more information on how to test the models and produce the above results, see the relative 
[models/README.md](./run/README.md).

## Cite
If you have any questions, please contact [alessio.monti@unimore.it](mailto:alessio.monti@unimore.it) or 
[alessia.bertugli@unimore.it](mailto:alessia.bertugli@unimore.it), or open an issue on this repo. 

If you find this repository useful for your research, please cite the following paper:
```bibtex
@proceedings{monti2020dagnet,
    title={DAG-Net: Double Attentive Graph Neural Network for Trajectory Forecasting},
    author={Alessio Monti and Alessia Bertugli and Simone Calderara and Rita Cucchiara},
    booktitle = {25th International Conference on Pattern Recognition (ICPR)},
    year={2020}
}
```
