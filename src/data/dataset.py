import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from data.texture_maps import process_texture

class CustomRegressionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 이미지 경로 추출
        texture_path = self.df.iloc[idx]['texture_path']
        normal_path = self.df.iloc[idx]['normal_path']
        height_path = self.df.iloc[idx]['height_path']
        
        # 정답 값 추출: 이제 roughness(1개)만 사용
        targets = torch.tensor(self.df.iloc[idx]['roughness'], dtype=torch.float32)

        # 이미지 로드
        texture_img = Image.open(texture_path).convert('RGB')
        normal_map = Image.open(normal_path).convert('RGB')
        height_map = Image.open(height_path).convert('RGB')

        # 변환 적용
        if self.transform:
            texture_img = self.transform(texture_img)
            normal_map = self.transform(normal_map)
            height_map = self.transform(height_map)

        return (texture_img, normal_map, height_map), targets

    
    
class CachedFeatureDataset(Dataset):
    def __init__(self, df, cache_root, split_name):
        self.df = df.reset_index(drop=True)
        self.split_cache = Path(cache_root) / split_name
        self.ids = [str(int(Path(p).stem)) for p in self.df['texture_path'].tolist()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        feat_path = self.split_cache / f"{sid}.pt"
        feat = torch.load(str(feat_path), weights_only=False).float()
        target = torch.tensor(self.df.iloc[idx]['roughness'], dtype=torch.float32)
        return feat, target
    
def _load_roughness_labels(csv_path):
    if not os.path.exists(csv_path):
        return {}

    labels_df = pd.read_csv(csv_path, header=None)
    if labels_df.shape[1] == 1:
        return {str(i + 1): float(labels_df.iloc[i, 0]) for i in range(len(labels_df))}

    if 'id' in labels_df.columns and 'roughness' in labels_df.columns:
        return labels_df.set_index('id')['roughness'].astype(float).astype(str).to_dict()

    if 'roughness' in labels_df.columns:
        roughness = labels_df['roughness'].astype(float).tolist()
        return {str(i + 1): roughness[i] for i in range(len(roughness))}

    return {str(i + 1): float(labels_df.iloc[i, 1]) for i in range(len(labels_df))} # second col


def _find_image_path(directory, sid, exts):
    for ext in exts:
        path = os.path.join(directory, f"{sid}{ext}")
        if os.path.exists(path):
            return path
    return None


def build_original_dataframe(base_dir="src/data/MOESM"):
    project_root = Path(__file__).resolve().parents[2]

    base_dir = Path(base_dir)
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir

    csv_path = base_dir / "ParticipantData.csv"
    texture_dir = base_dir / "MOESM1"

    train_ids_path = base_dir / "train_ids.csv"
    valid_ids_path = base_dir / "valid_ids.csv"

    label_map = _load_roughness_labels(str(csv_path))

    texture_files = [
        p for p in texture_dir.iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ]
    texture_files = sorted(
        texture_files,
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    def resolve_path(p):
        p = Path(p)

        if p.is_absolute():
            return p

        # 1순위: 현재 작업 디렉터리 기준
        cwd_path = Path.cwd() / p
        if cwd_path.exists():
            return cwd_path

        # 2순위: 프로젝트 루트 기준
        root_path = project_root / p
        if root_path.exists():
            return root_path

        # 아직 파일이 없더라도 프로젝트 루트 기준 경로로 반환
        return root_path

    rows = []
    for tex_path in texture_files:
        sid = tex_path.stem

        height_path, normal_path = process_texture(tex_path)

        height_path = resolve_path(height_path)
        normal_path = resolve_path(normal_path)

        if not normal_path.exists():
            raise FileNotFoundError(
                f"Normal map was not found for id={sid}: {normal_path}\n"
                f"process_texture()가 normal map을 생성했는지 확인하세요."
            )

        if not height_path.exists():
            raise FileNotFoundError(
                f"Height map was not found for id={sid}: {height_path}\n"
                f"process_texture()가 height map을 생성했는지 확인하세요."
            )

        roughness = label_map.get(sid)
        if roughness is None:
            try:
                roughness = label_map[str(int(sid))]
            except Exception:
                raise ValueError(
                    f"Could not find roughness label for sample id '{sid}' in {csv_path}"
                )

        rows.append({
            'id': str(int(sid)) if sid.isdigit() else sid,
            'texture_path': str(tex_path.resolve()),
            'normal_path': str(normal_path.resolve()),
            'height_path': str(height_path.resolve()),
            'roughness': float(roughness)
        })

    df_all = pd.DataFrame(rows)

    if train_ids_path.exists() and valid_ids_path.exists():
        train_ids = pd.read_csv(train_ids_path)['id'].astype(str).tolist()
        valid_ids = pd.read_csv(valid_ids_path)['id'].astype(str).tolist()

        df_train = df_all[df_all['id'].isin(train_ids)].reset_index(drop=True)
        df_valid = df_all[df_all['id'].isin(valid_ids)].reset_index(drop=True)

        used_ids = set(train_ids) | set(valid_ids)
        df_test = df_all[~df_all['id'].isin(used_ids)].reset_index(drop=True)
    else:
        df_train = df_all.reset_index(drop=True)
        df_valid = pd.DataFrame(columns=df_all.columns)
        df_test = pd.DataFrame(columns=df_all.columns)

    print(
        f"Loaded original dataframe: "
        f"all={len(df_all)}, train={len(df_train)}, valid={len(df_valid)}, test={len(df_test)}"
    )

    return df_train, df_valid, df_test


'''
def build_original_dataframe(base_dir="data/MOESM"):#original"):
    csv_path = os.path.join(base_dir, "ParticipantData.csv")
    texture_dir = os.path.join(base_dir, 'MOESM1')#'texture_image')

    label_map = _load_roughness_labels(csv_path)

    texture_files = [
        p for p in Path(texture_dir).iterdir()
        if p.suffix.lower() in {'.png', '.jpg'}
    ]
    texture_files = sorted(texture_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    rows = []
    for tex_path in texture_files:
        sid = tex_path.stem
        height_dir, normal_dir = process_texture(tex_path)
        
        roughness = label_map.get(sid)
        if roughness is None:
            try:
                roughness = label_map[str(int(sid))]
            except Exception:
                raise ValueError(
                    f"Could not find roughness label for sample id '{sid}' in {csv_path}"
                )

        rows.append({
            'texture_path': str(tex_path),
            'normal_path': normal_dir,
            'height_path': height_dir,
            'roughness': float(roughness)
        })

    return pd.DataFrame(rows)
'''


def build_dataframe(base_dir="data/MOESM"):
    csv_path = os.path.join(base_dir, "ParticipantData.csv")

    train_ids_path = os.path.join(base_dir, "train_ids.csv")
    valid_ids_path = os.path.join(base_dir, "valid_ids.csv")

    if not os.path.exists(train_ids_path):
        raise FileNotFoundError(f"train_ids.csv not found: {train_ids_path}")

    if not os.path.exists(valid_ids_path):
        raise FileNotFoundError(f"valid_ids.csv not found: {valid_ids_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ParticipantData.csv not found: {csv_path}")

    train_ids = pd.read_csv(train_ids_path)["id"].astype(str).tolist()
    valid_ids = pd.read_csv(valid_ids_path)["id"].astype(str).tolist()

    # ParticipantData.csv는 header가 없고, 100 x 4 형태임
    # 여기서는 첫 번째 column을 roughness GT로 사용
    labels_df = pd.read_csv(csv_path, header=None)

    def get_roughness_from_id(sid):
        idx = int(sid) - 1

        if idx < 0 or idx >= len(labels_df):
            raise IndexError(
                f"Sample id {sid} is out of range for {csv_path}. "
                f"CSV has {len(labels_df)} rows."
            )

        return float(labels_df.iloc[idx, 0])

    def build_from_id_list(id_list, split_name):
        rows = []

        tex_dir = os.path.join(base_dir, split_name, "texture_image")
        nor_dir = os.path.join(base_dir, split_name, "normal_map")
        hei_dir = os.path.join(base_dir, split_name, "height_map")

        for sid in id_list:
            tex = find_image_path(tex_dir, int(sid), [".png", ".jpg", ".JPG"])
            nor = find_image_path(nor_dir, int(sid), [".png", ".jpg", ".JPG"])
            hei = find_image_path(hei_dir, int(sid), [".png", ".jpg", ".JPG"])

            if not all([tex, nor, hei]):
                print(
                    f"[Warning] Missing image for id={sid}: "
                    f"texture={tex}, normal={nor}, height={hei}"
                )
                continue

            rough = get_roughness_from_id(sid)

            rows.append({
                "texture_path": tex,
                "normal_path": nor,
                "height_path": hei,
                "roughness": rough,
            })

        return pd.DataFrame(rows)

    train_df = build_from_id_list(train_ids, split_name="train")
    valid_df = build_from_id_list(valid_ids, split_name="valid")
    test_df = pd.DataFrame([])

    print(f"Loaded dataframe: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    return train_df, valid_df, test_df


'''
def build_dataframe(base_dir="data/MOESM"):
        csv_path = os.path.join(base_dir, "ParticipantData.csv")#adjective_rating_shuffled.csv")

        # If explicit train/valid id lists exist, use them
        train_ids_path = os.path.join(base_dir, 'train_ids.csv')
        valid_ids_path = os.path.join(base_dir, 'valid_ids.csv')
        if os.path.exists(train_ids_path) and os.path.exists(valid_ids_path):
            # read ids (assume header 'id')
            train_ids = pd.read_csv(train_ids_path)['id'].astype(str).tolist()
            valid_ids = pd.read_csv(valid_ids_path)['id'].astype(str).tolist()

            def build_from_id_list(id_list, split_name=None):
                rows = []
                for sid in id_list:
                    # sid corresponds to image filename without extension
                    # prefer split-specific directories (e.g. data/train/texture_image)
                    if split_name:
                        tex_dir = os.path.join(base_dir, split_name, 'texture_image')
                        nor_dir = os.path.join(base_dir, split_name, 'normal_map')
                        hei_dir = os.path.join(base_dir, split_name, 'height_map')
                    else:
                        tex_dir = os.path.join(base_dir, 'texture_image')
                        nor_dir = os.path.join(base_dir, 'normal_map')
                        hei_dir = os.path.join(base_dir, 'height_map')

                    tex = find_image_path(tex_dir, int(sid), ['.png', '.jpg', '.JPG'])
                    nor = find_image_path(nor_dir, int(sid), ['.png', '.jpg', '.JPG'])
                    hei = find_image_path(hei_dir, int(sid), ['.png', '.jpg', '.JPG'])
                    if all([tex, nor, hei]):
                        # labels CSV may not be present; try to read adjective_rating_shuffled.csv if available
                        if os.path.exists(csv_path):
                            labels_df = pd.read_csv(csv_path, header=None)
                            idx = int(sid) - 1
                            rough = labels_df.iloc[idx][0]
                        else:
                            rough = 0.0
                        rows.append({'texture_path': tex, 'normal_path': nor, 'height_path': hei, 'roughness': rough})
                return pd.DataFrame(rows)

        # build using split subfolders if present
        train_df = build_from_id_list(train_ids, split_name='train')
        valid_df = build_from_id_list(valid_ids, split_name='valid')
        test_df = pd.DataFrame([])
        return train_df, valid_df, test_df
'''


def find_image_path(directory, idx, exts):
    """확장자 순회하며 실제 파일 경로 탐색"""
    for ext in exts:
        path = os.path.join(directory, f"{idx}{ext}")
        if os.path.exists(path):
            return path
    return None