# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from dynamic_stereo.scripts.download_utils import build_arg_parser, download_dataset


DEFAULT_LINK_LIST_FILE = os.path.join(os.path.dirname(__file__), "links.json")
DEFAULT_SHA256S_FILE = os.path.join(os.path.dirname(__file__), "dr_sha256.json")


if __name__ == "__main__":
    parser = build_arg_parser(
        "dynamic_replica", DEFAULT_LINK_LIST_FILE, DEFAULT_SHA256S_FILE
    )

    args = parser.parse_args()
    os.makedirs(args.download_folder, exist_ok=True)
    download_dataset(
        str(args.link_list_file),
        str(args.download_folder),
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        download_splits=args.download_splits,
        checksum_check=bool(args.checksum_check),
        clear_archives_after_unpacking=bool(args.clear_archives_after_unpacking),
        sha256s_file=str(args.sha256_file),
        skip_downloaded_archives=not bool(args.redownload_existing_archives),
    )
