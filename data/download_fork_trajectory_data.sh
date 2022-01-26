#!/bin/bash
# download the fork trajectory data from https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/8TTXZ7/IDCWI2&version=15.0
# 
# @incollection{DVN/8TTXZ7/IDCWI2_2018,
# author = {Bhattacharjee, Tapomayukh and Song, Hanjun and Lee, Gilwoo and Srinivasa, Siddhartha S.},
# publisher = {Harvard Dataverse},
# title = {subject10_banana_wrenches_poses.tar.gz},
# booktitle = {A Dataset of Food Manipulation Strategies},
# year = {2018},
# version = {V15},
# doi = {10.7910/DVN/8TTXZ7/IDCWI2},
# url = {https://doi.org/10.7910/DVN/8TTXZ7/IDCWI2}
# }

urlbase=https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/8TTXZ7
echo "Downloading fork trajectory dataset."
foldername=fork_trajectory_banana
if [[ ! -d "$foldername" ]]; then
  mkdir $foldername 
  wget -O $foldername/subject1_banana_wrenches_poses.tar.gz $urlbase/WFLYTR
  wget -O $foldername/subject2_banana_wrenches_poses.tar.gz $urlbase/QA8MKX
  wget -O $foldername/subject3_banana_wrenches_poses.tar.gz $urlbase/VWXQQM
  wget -O $foldername/subject4_banana_wrenches_poses.tar.gz $urlbase/ANDMQH
  wget -O $foldername/subject5_banana_wrenches_poses.tar.gz $urlbase/DJJIS8
  wget -O $foldername/subject6_banana_wrenches_poses.tar.gz $urlbase/FK7YR3
  wget -O $foldername/subject7_banana_wrenches_poses.tar.gz $urlbase/JJEUAF
  wget -O $foldername/subject8_banana_wrenches_poses.tar.gz $urlbase/GEHSTD
  wget -O $foldername/subject9_banana_wrenches_poses.tar.gz $urlbase/QTFS0F
  wget -O $foldername/subject10_banana_wrenches_poses.tar.gz $urlbase/IDCWI2
  wget -O $foldername/subject11_banana_wrenches_poses.tar.gz $urlbase/YMIX7F
  wget -O $foldername/subject12_banana_wrenches_poses.tar.gz $urlbase/DWYXC5
  echo "Downloading fork trajectory completed!"
  pushd $foldername
  for id in {1..12}; do
    tar -xzf subject${id}_banana_wrenches_poses.tar.gz
  done;
  popd
fi

echo "Downloading fork trajectory dataset for carrot."
foldername=fork_trajectory_carrot
if [[ ! -d "$foldername" ]]; then
  mkdir $foldername 
  wget -O $foldername/subject1_carrot_wrenches_poses.tar.gz  $urlbase/CG2EJL
  wget -O $foldername/subject2_carrot_wrenches_poses.tar.gz  $urlbase/CNW32N
  wget -O $foldername/subject3_carrot_wrenches_poses.tar.gz  $urlbase/EFTBBD
  wget -O $foldername/subject4_carrot_wrenches_poses.tar.gz  $urlbase/C0E51G
  wget -O $foldername/subject5_carrot_wrenches_poses.tar.gz  $urlbase/RVCHMM
  wget -O $foldername/subject6_carrot_wrenches_poses.tar.gz  $urlbase/PGZK3C
  wget -O $foldername/subject7_carrot_wrenches_poses.tar.gz  $urlbase/O1RWFG
  wget -O $foldername/subject8_carrot_wrenches_poses.tar.gz  $urlbase/OO4N50
  wget -O $foldername/subject9_carrot_wrenches_poses.tar.gz  $urlbase/BKPPSS
  wget -O $foldername/subject10_carrot_wrenches_poses.tar.gz $urlbase/IALDDC
  wget -O $foldername/subject11_carrot_wrenches_poses.tar.gz $urlbase/G3QEYQ
  wget -O $foldername/subject12_carrot_wrenches_poses.tar.gz $urlbase/H3K4DX
  echo "Downloading fork trajectory completed!"
  pushd $foldername 
  for id in {1..12}; do
    tar -xzf subject${id}_carrot_wrenches_poses.tar.gz
  done;
  popd
fi

echo "Downloading fork trajectory dataset for carrot."
foldername=fork_trajectory_strawberry
if [[ ! -d "$foldername" ]]; then
  mkdir $foldername 
  wget -O $foldername/subject1_strawberry_wrenches_poses.tar.gz  $urlbase/VVDLAV
  wget -O $foldername/subject2_strawberry_wrenches_poses.tar.gz  $urlbase/7XDEJK
  wget -O $foldername/subject3_strawberry_wrenches_poses.tar.gz  $urlbase/UU91LF
  wget -O $foldername/subject4_strawberry_wrenches_poses.tar.gz  $urlbase/3RPDYI
  wget -O $foldername/subject5_strawberry_wrenches_poses.tar.gz  $urlbase/MKD4XG
  wget -O $foldername/subject6_strawberry_wrenches_poses.tar.gz  $urlbase/67IMHB
  wget -O $foldername/subject7_strawberry_wrenches_poses.tar.gz  $urlbase/TVMDQW
  wget -O $foldername/subject8_strawberry_wrenches_poses.tar.gz  $urlbase/JJHWVV
  wget -O $foldername/subject9_strawberry_wrenches_poses.tar.gz  $urlbase/JWU3TA
  wget -O $foldername/subject10_strawberry_wrenches_poses.tar.gz $urlbase/LDU4D8
  wget -O $foldername/subject11_strawberry_wrenches_poses.tar.gz $urlbase/7EHRFI
  wget -O $foldername/subject12_strawberry_wrenches_poses.tar.gz $urlbase/MVUML5
  echo "Downloading fork trajectory completed!"
  pushd $foldername 
  for id in {1..12}; do
    tar -xzf subject${id}_strawberry_wrenches_poses.tar.gz
  done;
  popd
fi
