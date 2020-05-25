## Pre-trained models

This directory contains the pre-trained models used to obtained the numbers in the paper. Because of some changes/fixes 
during the code refactoring, we obtained different numbers - almost always better :) - from the ones in the original 
paper:

| <td colspan=2 style="text-align:center; font-weight:bold">BSK Offense<td colspan=2  style="text-align:center; font-weight:bold">BSK Defense <td colspan=2  style="text-align:center; font-weight:bold">Stanford Drone
| --- | --: | --: | --: | --: | --: | --: | --: 
| <td colspan=1 style="text-align:center; font-weight:bold">ADE <td colspan=1 style="text-align:center; font-weight:bold">FDE <td colspan=1 style="text-align:center; font-weight:bold">ADE <td colspan=1 style="text-align:center; font-weight:bold">FDE <td colspan=1 style="text-align:center; font-weight:bold">ADE <td colspan=1 style="text-align:center; font-weight:bold">FDE 
| VRNN | 9,41 | 15,56 |  7,16 | 10,50 | 0,58  | 1,17 |
| A-VRNN | 9,48 | 15,52 | 7,04  | 10,34 | 0,56  | 1,14 |
| DAG-Net | <em>8,98</em> | <em>14,08</em> | <em>6,87</em> | <em>9,76</em> | <em>0,53</em>  | <em>1,04</em> |

To evaluate again these models or use them as a starting point for further training, move the relative experiment in the
correct `/runs/<model>` directory and run the relative script (see [models/README.md](./models/README.md) for more info).
