## Training Model

```bash
python methods/method_ladeda.py \
  --data-root "D:/CelebDF_Frames_Split_Correct" \
  --out-dir outputs_ladeda \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-4 
```
## Run Backend

```bash
cd backend 
python app.py 
```
## Run Frontend

```bash
cd frontend 
npm start 
```
