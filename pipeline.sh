#!/bin/bash

echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `nii2npy`(label)...';
cargo run --release --package detection -- nii2npy -D ../../dataset --target label 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `nii2npy`(scan)...';
cargo run --release --package detection -- nii2npy -D ../../dataset --target scan 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `rnpy2unpy`...'
cargo run --release --package detection -- rnpy2unpy -D ../../dataset 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `rnpy2png`...'
cargo run --release --package detection -- rnpy2png -D ../../dataset 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `unpy2png`...'
cargo run --release --package detection -- unpy2png -D ../../dataset 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `ct-window`...'
cargo run --release --package detection -- ct-window -D ../../dataset --centre 60 --width 200 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `rnpy2unique`...'
cargo run --release --package detection -- rnpy2unique -D ../../dataset 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `paper-algos2npy`...'
cargo run --release --package detection -- paper-algos2npy -D ../../dataset --ranges 0-9 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `algo-bench`...'
cargo run --release --package detection -- algo-bench -D ../../dataset --save-npy --save-png 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
# shellcheck disable=SC2016
echo 'Running command `eroded-coe`...'
cargo run --release --package detection -- eroded-coe -D ../../dataset 2>&1 | tee -a log.txt;
echo '------------------------------------------------------------------'
echo 'All finished.'