# -*- coding: utf-8 -*-
import requests
import os
import tqdm
from clint.textui import progress


if __name__ == "__main__":
    """
        ref: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo
    """
    model_url_map = {
        "backbone_r100.pth": "https://public.dm.files.1drv.com/y4m1mOLIYEhwVVNgTtd-GOb-WGh09dBU-ok9k8Io2yEN7avQZMP31v_oJJA4SEJCbc1N5sr0UVLhxcojq4fimFJPzj8-Ft8kTNTQfA2PK6_9itFKVAFXYRVIXdl4RNje1GrdS5jk8WiMN5xcjoLLWVd_SreM_1YhiMtNUaCdbqDFPwmg2-hNg_152g2oc-sliKGFO10R7Z-6c785BcYQBSAosfUVfoWn7xSG0WUIK_k0Hc"
    }

    out_dir = "./models/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for model_path, model_url in tqdm.tqdm(model_url_map.items()):
        urlData = requests.get(model_url, stream=True)
        out_file = out_dir + os.path.basename(model_path)

        with open(out_file, 'wb') as f:
            print("Download", out_file)
            total_length = int(urlData.headers.get('content-length'))
            for chunk in progress.bar(urlData.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
