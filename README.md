# CPWS

CPWS: Generating High-Quality Data from Confident Programmatic Weak Supervision

**Framework**
<img src="framework.png"/>

**Dependencies**

Create virtual environment:

```
conda env create -f env.yml
source activate cpws
```

**Data**

Datasets are available here:

Yelp: [link](https://drive.google.com/drive/folders/1rI6wKit4oq3nneqyw4uWrvKw7_b3ut4r?usp=drive_link)

AGNews: [link](https://drive.google.com/drive/folders/1IFuRObRwPBLjTdFgjKxzbyR5Z0KxYxHo?usp=drive_link)

Tennis Rally: [link](https://drive.google.com/drive/folders/1z983x_QPvDwJqLaWxevSQ9xRHmJenrBa?usp=drive_link)

Basketball: [link](https://drive.google.com/drive/folders/1Z7Odq8RukYWYkXFEXB9pWD7Td77miLb2?usp=drive_link)

Our project is built based on Python, Pytorch and Wrench. We sincerely thank the efforts of all the researchers!

**Examples**

Train an end model on true data-label pairs in a supervised manner.

```
python supervised_learning.py
```

Train an end model on PWS-generated weak datasets.

```
python weakly_supervised_learning.py
```

Split dataset into $X^{gt}$ and $X^u$.

```
from cpws import CPWS

cpws = CPWS()

ds_v = cpws.load_json("datasets/yelp/train.json")
len_train = len(ds_v)
print(len_train)

AR = 0.01
rd, ud = cpws.data_split(ds_v, AR)
cpws.save_json(rd, 'datasets/yelp/datasets_1/train_rd.json')
cpws.save_json(ud, 'datasets/yelp/datasets_1/train_ud.json')
```

Weak dataset preparing
```
ds_v_ = cpws.load_json('datasets/yelp/datasets_1/train_ud.json')
weak_datasets = cpws.weak_data_gen_full(ds_v_, fd_name='yelp')
for i in range(len(weak_datasets)):
    cpws.save_json(weak_datasets[i], 'datasets/yelp/datasets_1/train_ud_w_ab%d.json'%i)
```

**Reference**

- [WRENCH: A Comprehensive Benchmark for Weak Supervision](https://arxiv.org/abs/2109.11377) [[code]](https://github.com/JieyuZ2/wrench)
