# pictollms/data/image_processor.py
import os
import lmdb
import numpy as np
import torch
from typing import Dict, Optional, Tuple

class ImageProcessor:
    
    def __init__(self, lmdb_path: str, resolution: int = 224):    # config parameters
        

        self.lmdb_path = lmdb_path
        self.resolution = resolution
        self.env = None
        
        if os.path.exists(lmdb_path):
            self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    def get_image(self, picto_id: int) -> Optional[torch.Tensor]:

        if self.env is None:    
            return self._get_empty_image()
        
        with self.env.begin() as txn:
            key = f"picto_{picto_id}".encode()
            data = txn.get(key)
            
            if data is None:
                return self._get_empty_image()
            
            img_array = np.frombuffer(data, dtype=np.float32)     # image conversion handlers
            img_array = img_array.reshape(self.resolution, self.resolution, 3)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]    check format, 
            
            return img_tensor
    
    def _get_empty_image(self) -> torch.Tensor:
        return torch.zeros(3, self.resolution, self.resolution)
    
    def get_batch_images(self, picto_ids: list) -> torch.Tensor:
        images = []
        for picto_id in picto_ids:
            img = self.get_image(picto_id)
            images.append(img)
        
        return torch.stack(images)
    
    def close(self):    # check lmdb docs for reference
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def __del__(self):
        self.close()