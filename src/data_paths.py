import tensorflow as tf
from pathlib import Path
from typing import List, Literal

SplitName = Literal["Train", "Validation", "Test"]

def get_video_dir(data_root: Path, 
                  video_id: int, 
                  split: SplitName
                  ) -> Path:
    #returneaza folderul care contine frame urile unui video
    return data_root / split / str(int(video_id))

def get_frame_paths(data_root: Path, 
                    num_frames: int, 
                    video_id: int, 
                    split: SplitName, 
                    strict: bool = True
                    ) -> List[Path]:
    #returneaza lista de frame uri sortate pentru un video_id"
    vdir = get_video_dir(data_root, video_id, split)

    if not vdir.exists():
        msg = f"nu există folderul pentru video_id={video_id} în {split}: {vdir}"
        if strict:
            raise FileNotFoundError(msg)
        return []


    frame_paths = sorted(vdir.glob("*.jpg"))

    if strict:
        if len(frame_paths) != num_frames:
            raise ValueError(
                f"{split}/video_id={video_id}: așteptam {num_frames} cadre, am găsit {len(frame_paths)}"
            )

        # verificare suplimentară: numele să fie exact 00001.jpg ... 00037.jpg
        expected = [f"{i:05d}.jpg" for i in range(1, num_frames + 1)]
        actual = [p.name for p in frame_paths]
        if actual != expected:
            raise ValueError(
                f"{split}/video_id={video_id}: numele cadrelor nu sunt in formatul asteptat.\n"
                f"expected: {expected[:5]} ...\n"
                f"actual:   {actual[:5]} ..."
            )

    return frame_paths

def make_frame_path_tf(
    data_root: tf.Tensor, 
    #num_frames: int, 
    split: tf.Tensor, 
    video_id: tf.Tensor, 
    frame_idx: tf.Tensor
) -> tf.Tensor:
    # data/<split>/<video_id>/<00001.jpg>
    video_id = tf.cast(video_id, tf.int32)
    f = tf.strings.as_string(frame_idx, width=5, fill='0')
    fname = tf.strings.join([f, ".jpg"])
    # join cu separator "/"
    return tf.strings.join([data_root, split, tf.strings.as_string(video_id), fname], separator="/")