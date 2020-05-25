## Pre-trained models

This directory contains the pre-trained models used to obtained the numbers in the paper. Because of some changes/fixes 
during the code refactoring, we obtained different numbers - almost always better :) - from the ones in the original 
paper:

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

To evaluate again these models or use them as a starting point for further training, move the relative experiment in the
correct `/runs/<model>` directory and run the relative script (see [models/README.md](./models/README.md) for more info).
