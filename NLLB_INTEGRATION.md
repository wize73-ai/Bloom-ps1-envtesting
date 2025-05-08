# NLLB Translation Model Integration

This document describes the integration of the NLLB-200-1.3B model as the primary translation model in CasaLingua, replacing MBART and providing improved translation quality and MPS compatibility on Apple Silicon devices.

## Overview

NLLB (No Language Left Behind) is a breakthrough machine translation AI model developed by Meta that supports 200 languages within a single model. The 1.3B parameter variant provides a good balance between high-quality translations and resource efficiency, making it suitable for use on a wide range of hardware including Apple Silicon devices.

## Changes Made

The integration involved the following changes:

1. **Updated Model Registry**: Configured `facebook/nllb-200-1.3B` as the primary translation model in the system, replacing the previous MBART model.

2. **Fixed Device Selection Logic**: Modified the device selection logic to allow NLLB models to run on MPS (Metal Performance Shaders) when available on Apple Silicon devices.

3. **Enhanced Language Code Handling**: Updated the language code mapping to support the NLLB-specific language codes for all 200 languages.

4. **Added NLLB-Specific Wrapper Support**: Modified the translation model wrapper to properly handle NLLB models, including support for language code conversion and optimized generation parameters.

5. **Created Test Scripts**: Implemented test scripts to verify that NLLB translations, especially Spanish to English translations, work correctly on Apple Silicon devices.

## Implementation Files

- **`update_to_nllb_13b.py`**: Main script that updates the model registry to use NLLB-200-1.3B as the primary translation model.

- **`fix_nllb_mps_compatibility.py`**: Script that fixes the device selection logic in the ModelLoader to allow NLLB models to run on MPS.

- **`update_nllb_language_codes.py`**: Script that updates the language code handling to support the NLLB-specific language codes.

- **`test_nllb_translation.py`**: Script that tests NLLB translations with a focus on Spanish to English translations.

- **`scripts/nllb_language_lookup.py`**: Utility script for looking up NLLB language codes for all supported languages.

## Installation

To install and activate the NLLB integration, follow these steps:

1. **Update the Model Registry**:
   ```bash
   python update_to_nllb_13b.py
   ```

2. **Fix MPS Device Compatibility**:
   ```bash
   python fix_nllb_mps_compatibility.py
   ```

3. **Update Language Code Handling**:
   ```bash
   python update_nllb_language_codes.py
   ```

4. **Restart the Application**:
   Restart your application to ensure that the changes take effect.

5. **Test the Integration**:
   ```bash
   python test_nllb_translation.py
   ```

## Supported Languages

NLLB-200 supports 200 languages, including:

- English (eng_Latn)
- Spanish (spa_Latn)
- French (fra_Latn)
- German (deu_Latn)
- Italian (ita_Latn)
- Portuguese (por_Latn)
- Chinese (Simplified - zho_Hans, Traditional - zho_Hant)
- Japanese (jpn_Jpan)
- Arabic (arb_Arab)
- Russian (rus_Cyrl)
- And many more...

To see the full list of supported languages, run:
```bash
python scripts/nllb_language_lookup.py --all
```

To look up a specific language code, run:
```bash
python scripts/nllb_language_lookup.py [language_code or language_name]
```

## MPS Compatibility

The integration specifically enhances MPS compatibility for NLLB models on Apple Silicon devices. The device selection logic has been modified to:

1. Allow NLLB models to run on MPS
2. Continue forcing MBART models to run on CPU when on MPS devices
3. Optimize translation performance on Apple Silicon

## Performance Considerations

- **Memory Usage**: NLLB-200-1.3B requires approximately 8GB of RAM to run efficiently.
- **Speed**: On Apple Silicon MPS, translation speed is significantly improved compared to running on CPU.
- **Quality**: NLLB-200-1.3B provides higher quality translations, especially for low-resource languages, compared to MBART and MT5.

## Troubleshooting

If you encounter issues with the NLLB integration:

1. **Verify Model Downloads**: Ensure that the NLLB-200-1.3B model has been downloaded.
2. **Check Memory**: Ensure that your system has sufficient memory (8GB+ recommended).
3. **Verify Device Selection**: Use the `test_nllb_translation.py` script to verify that the model is using the correct device.
4. **Check Language Codes**: If translations fail for specific languages, verify the language codes using the `nllb_language_lookup.py` utility.

## References

- [NLLB Research Paper](https://arxiv.org/abs/2207.04672)
- [NLLB Models on Hugging Face](https://huggingface.co/facebook/nllb-200-1.3B)
- [FLORES-200 Benchmark](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)