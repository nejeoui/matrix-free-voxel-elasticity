# Redactions in the public artifact

Every file is published as it was recorded, with one exception: the per-instance cloud API keys returned by the GPU provider are replaced by `"REDACTED"`. Those keys were credentials, and the instances were destroyed after collection. Nothing else in these files is changed, and no measured value is affected.

Each entry gives the SHA-256 of the original file and of the published file. The originals are available on request and can be checked against these values.

| file | original SHA-256 | published SHA-256 |
|---|---|---|
| `results/session-b-20260923/create-response.json` | `326759a428dd65ee4195949e3559ab5e00896a08bba7039c381a509b33417b0b` | `d96d114fa56f3b586d6cccb51d7813692df8d884bd807d434bcabb7750ec332f` |
| `results/session-c-20260923/create-response.json` | `982796400c23febd5576d390453277618a0cf3868864e96068f0d48031c40f7b` | `bec008aee3c1022413123014c141e54d0de80e9d381e1fd4a1a0786efb84b07b` |
| `results/session-d-20260923/create-response.json` | `65f9f3d34280254e8263e4b801d7f7057aa9b28622dc8f23efe965d6e3b05d80` | `63ed5a20b31cfd4f36ad9b0123406ec4acf40aac32bae944e7b707dbe3ba3fa0` |
| `results/session-e-host1-20260925/create-response.json` | `c2bce53d4af16a36e61d4f8dc37dafdd0f66c997a5078f58ae74abe836efa596` | `56b31dd9e9de3c5d706ad51651bdb082dc747cdebe1e14f72d02376c0825fbcd` |
| `results/session-e-host2-20260925/create-response.json` | `1df29a8650ec43718a477382f7acd4223e7d6d996f18ca8b96167aeec89ac641` | `19a8da8a88e5de9cf32b591a9fc672640b2f85a0a174ed2b97e29d772f852ad1` |
| `results/session-f-a100-20260926/create-response.json` | `d1f7a9ee641b57cbec9268d6f75298b4b0e65803906a2419f20199b65d9f7afb` | `95606f35ba620c26e942169afebe2da4cfffccf86171fab989002fa94816d4ec` |
| `results/session-f-h100-20260926/create-response.json` | `368e7ce4174e4a5546204c754e5b9273eb959dd184cdfbef67b4c73b370fd82f` | `36ed9ed0711b3d35470a9d0649a8151fdd0b5dd7d529a70e83c492e8e4afcdd9` |
| `results/session-f-h100-20260926-attempt02/create-response.json` | `c5456f8a365a5d36d04c49da4ff59ab2aceddf788f25c048004bd8c1e1f8e747` | `29500700232dcb8465f6f3f0bb1a26ca7c843bc40715c928471bded4308c2a0d` |
| `results/session-f-rtx4090-20260926/create-response.json` | `7ab7537d4d075f33017859b682c98ab4add39289b3269eba752b95eb7db314a2` | `9ec26e529a574d81737de157dfb352e8a86b81e65ba70ba440a9000465929c6e` |
| `results/session-g-host1-20260926/create-response.json` | `274f377625d9710e29d872aa72bd8ed6d4eda9af34fff91d958d8c7fbe732a58` | `2f5df97e214b8565fb48d354eee66d13ac0b4034de7630e93a2ebd2d89d19744` |
| `results/session-g-host2-20260926/create-response.json` | `694a625e58691bceceefa5ff526a8f7a75e1ceaada8d892f8273799bfdaa5f69` | `1c3a4df8b7732641ada5cb141bf1962a0232950cf7a85e31eaa407a4abd6eccf` |
| `results/session-h-host1-20260926/create-response.json` | `13d9eb32466be97218df6be5c11a7605196b19feea6c5e26f485aa5e701652e3` | `e484f4eaa40a547709954abb1e87a88926444bd79f802a42e42394214bc103cc` |
| `results/session-h-host2-20260926/create-response.json` | `03c5b8d7103dab3eca9a7209576daf1d5cbe504eac99f8e29baf769778fb7a37` | `ae1ccc32bac1440b5d6d150bc8efcb5f6fc49500a1287833e429967f8a6cd222` |
| `results/session-i-a100-20260926/create-response.json` | `06cd79c6ded7d462f6ee7b7fbb8ba95674734b4795c3fe2cbb91a25883c1cd71` | `65b5e0d13742c0e6c785039788072842ef73510c79ebf0d2d9255f2f6bc69c27` |
| `results/session-i-h100-20260926/create-response.json` | `3aa4568a3e2ebd0c333127a3031809d85c98b32ef792e86dc4e4a800de6f0100` | `6df5832ef5ce1527bf76eae50f669e9a3ca4750f04a990061c8bebf02c74cbeb` |
| `results/session-i-rtx4090-20260926/create-response.json` | `a76ba905f6363227c4a6e29126c341274888d7354e95f6a01f4a87a53d7e8e2d` | `66cb8438411176a38d4ea6010d636bab90f68532a7f400f9d2bb7dc01bc4d343` |
| `results/session-j-rtx4090vm-20260926/create-response.json` | `737d81186239a9c64645ab8c4b93f0ec5a14d68b9c8576308116e2e818b31db4` | `fc3e615451f0cdd956f40754197be7de49d51d9210e425039f4f0067c842638f` |
| `results/session-j-rtx4090vm-20260926-attempt02/create-response.json` | `197abf1fd8296e9706f71b7889dcd522cc693167b2c1be497ab9d87bee1477a4` | `91d3b08664232140245f027c4c84d610919e19375561c0d8571ca06c08f05e1d` |
| `results/session-j-rtx4090vm-20260926-attempt03/create-response.json` | `af38585cf127097d9c58e314e0b9dd66c5e003784fc8f5b8b5a17a2a5f5ad776` | `0619b86217d6645de18a7762f8dea31f4fb4451c870e5f721979b31e776fc40c` |
| `results/session-j-rtx4090vm-20260926-attempt04/create-response.json` | `917b27cd701b05f2d3bc97a4eccfd5e526d834aa5716a09bdffd2a28fd9107dc` | `e8ffdb830924b2d1098dfb88516faf8f99685c235e30d02e4380c6bdde3011f5` |
| `results/session-j-rtx4090vm-20260926-attempt05/create-response.json` | `96c7fa35b488d2378999a29f1be7a625ff292b87f2eb3f6ec570b8a47ab36780` | `abc7e1f78f2771c506fb6f955b3a710c1ed550b04ebb405df371e8ca99c97e8c` |
| `results/session-j-rtx4090vm-20260926-attempt06/create-response.json` | `a1ba80d9ae76a6d0a9292266c0d0292c9c1277b7fa6669d59c0f1816193e1b9f` | `8e98b38fd28b1fdcfd4b895f5f47b973a755222b3ec3d2cf7b5368140e0f5515` |
| `results/session-j-rtx4090vm-20260926-attempt07/create-response.json` | `ed0304a9f5fef9eb5d4026ee4ae1ae822e9eb2febbb5c4599033892cece9a40a` | `1db6c524062d917f428dcae86784faa8989d7426c0ffa4726f7ac6e76f62d066` |
