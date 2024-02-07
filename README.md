# tfop_code
code for analysing MuSCAT1/2/3/4 photometry data for TFOP

## Installation

```shell
$ conda create -n tfop python=3.11
$ conda activate tfop
(tfop) $ python -m pip install -r https://raw.githubusercontent.com/jpdeleon/tfop_code/main/requirements.txt
(tfop) $ python -m pip install -e git+https://github.com/jpdeleon/tfop_code.git#egg=tfop-code
(tfop) $ python -m pip install jupyterlab
```
Then,
```python
from tfop_analysis import Star, Planet, LPF
```

