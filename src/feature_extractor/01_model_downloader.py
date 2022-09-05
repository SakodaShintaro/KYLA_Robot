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
        "backbone_ms1mv3_r100.pth": "https://public.dm.files.1drv.com/y4m5xWt2ySHE0aN_KdKC1zkadNUZYIX0GdEDegZCsEHWE2g8qJ3eN2BYm3_emPrxdZIxvItGKh-S2NdG9wuwqzC2TVWwKwnl2kFpfDHeNmzf7QVTKYBgEFfzaUFyzw9LEsecWfOZEQGVfn6HCPYPhCpn6tw29-6mu9tzFKn8epr0_Nlvya5vP5l2KtcBkXUW8iNMM1gMgqwgW7uDN5-czIbEw8BfhQtW9lsjDSJ6X9p0Jo",
        "backbone_glint360k_r100.pth": "https://public.dm.files.1drv.com/y4mfmVZXbM3nTaxhhjX9uL-lJTNsXNygeIYuGHL-_n9YqM25w702aCyu1HwHQLwmASP9HyLxC4E8adl2kEDMx36RZqWGx7HfZyXLf6Lt9zBlk-EfLOms70DJGJs3CUKCgjQ9coqC6c9g7MNoL_i5aGW9e4D1J-pU4jOxxXKjUcqYabJeXJI2FdxDVJwayy8PqSc_qTmdj1vMbqon_eaTb9XVpWRLk2pUPvJcDoyz4dUHH0"
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
