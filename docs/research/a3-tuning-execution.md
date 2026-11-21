# A3 tuning: dang ky thuc thi 21/11/2026

Chu track: Gia Bao, nhanh `giabao`. Yeu cau moi cua chu repo: mo Kaggle trong Codex va bat dau chay nhieu huong tuning, khong chi lap ke hoach. Day la quyen trien khai P1-P3 cua [ke hoach](a3-tuning-plan.md), khong thay nhiem vu/han cua Duc hoac nghiem thu cua Tung. Khong huong dan bat lai PoseLab.

## Khoa truoc outer-test

- Giu nguyen 6 recipe C0-C5, 3 model A3/TCN/GRU, 3 seed 42/142/242, 4 outer va 3 inner; 648 inner fits + 36 refits. C0-C3 kiem LR/regularization, C4 lich cosine, C5 smoothing. Khong them kien truc/search theo outer scores.
- `training/a3_tuning/runner.py` import model/helper goc, khong sua nguon cua Tung/Duc. Nguong chinh 0.5; chon recipe bang pooled inner accuracy, balanced accuracy, CE tu logits, thu tu C0-C5. Outer score chi duoc tinh sau selection.json da ghi.
- Chi xuat policy fixed trong run nay. Nguong balanced/sensitivity va timing la phan bo sung chua chay; khong goi model tot hon toan dien chi tu Accuracy.
- Lam ro de tai lap C0: epoch selection giu dung baseline voi unweighted unsmoothed CE tu score clip [1e-7, 1-1e-7], chi cap nhat neu giam hon 1e-6. Recipe tie-break van dung CE log-softmax khong clip. Cosine T_max=50 ca inner/refit, step sau optimizer, khong AMP/ensemble/TTA.
- Tat ca model cung runtime Kaggle, hai GPU T4 cung loai; moi fit dung mot GPU, khong DataParallel. Moi GPU duoc chia deu 6 fold/model. Hai CPU thread/process; FP32 va cudnn deterministic theo helper goc. Khong sua/cai lai CUDA/PyTorch he thong; runtime thuc dung ghi trong lock, khong goi la tai lap runtime RTX3050 cua Tung.
- Full fingerprint khoa recipes, nguon code/design, input SHA, runtime va budget; full lock them smoke SHA va so worker. Smoke bat buoc du lieu that, 9 inner fits (3 model x C0/C4/C5, 1 epoch) va 3 refit/outer-inference smoke, **12 fits ngoai ngan sach full**. Unit tests dung synthetic, khong thay smoke.
- Checkpoint co model, optimizer, scheduler, best state, epoch/history, RNG Python/NumPy/Torch/CUDA, loader generator va augmentation RNG. Ghi moi 5 epoch va luc fit ket thuc; interruption co the tinh lai toi da 4 epoch chua luu, khong doi recipe. Artifact hoan tat kiem SHA truoc resume; lock khac phai output moi.
- Khong dung gioi han mot gio de tu cat seed/fold/model. Sau mot gio co the chi co tien do/partial; khong tuyen bo FULL khi chua du 684 fits, 36 outer, 1440 OOF fixed. Khong hua thang. Khong tu thue GPU hay mua quota.

## Lenh Kaggle

Notebook tai code tu commit Git bat bien va kiem SHA tung file; tai 4 input chuan tu public GitHub bang tool da co, doi chieu hash, khong upload video rieng. Chay regression tests truoc smoke; smoke GPU phai PASS truoc full. Launcher luu log/stdout trong `/kaggle/working/a3-tuning-20261121/`.

```bash
python -m training.a3_tuning.runner --mode smoke --device cuda --input-root data/gmdcsa24 --output /kaggle/working/a3-tuning-20261121/smoke
python -m training.a3_tuning.runner --mode full --device cuda --workers 2 --input-root data/gmdcsa24 --smoke-evidence /kaggle/working/a3-tuning-20261121/smoke --output /kaggle/working/a3-tuning-20261121/full
```

Dung Save & Run All de chay nen va luu output khi Kaggle hoan tat; giu notebook private, khong thay quyen chia se. `progress.json` va `worker0.log`/`worker1.log` cho biet so fits/fold da xong. `predictions_partial.csv`/`metrics_partial.csv` la **partial**, khong so truc tiep so tong hop full cua Tung. `completion.json` chi xuat sau du inventory. Timing va nghiem thu khoa hoc doc lap chua duoc thay bang viec training ket thuc.

Neu co loi: giu log/checkpoint, ghi blocker, sua tren output/lock moi neu doi code/config; khong gia so, khong bo guard. Thu thap archive ket qua khi xong de review, chua tu thay model/deployment hoac ghi de results cu. Hien trang chay thuc te theo README/bao cao Gia Bao va Kaggle version, khong suy tu ban dang ky nay.
