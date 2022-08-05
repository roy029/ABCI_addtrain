# module load gcc/9.3.0 python/3.8 cuda/11.3 cudnn/8.2.4
python3 Testfile/tsv2txt.py /home/acd13734km/Retake/Git/git/data2022/result/result_codeT5_kogi_test.tsv
python3 calc_code_bleu.py --refs Testfile/reff.txt --hyp Testfile/hypp.txt --lang python --params 0.40,0.40,0.10,0.10