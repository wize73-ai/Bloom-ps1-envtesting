# CasaLingua Demo

This demo showcases the key capabilities of the CasaLingua language AI platform in an interactive terminal application.

## Features Demonstrated

The 2-minute demo cycles through these key features:

1. **System Health Check**: Displays loaded models and API connectivity status
2. **Translation**: Demonstrates high-quality translation using the MBART model
3. **Text Simplification**: Shows how complex text can be simplified to various reading levels
4. **Veracity Auditing**: Demonstrates translation verification and quality assessment

## Running the Demo

### Option 1: Run in a separate terminal (recommended)

```bash
# Make the script executable (first time only)
chmod +x scripts/run_demo.sh

# Run the demo
./scripts/run_demo.sh
```

This will:
- Check if the CasaLingua API is running (starts it if needed)
- Install required dependencies
- Launch the interactive demo

### Option 2: Run directly

If you prefer to run the demo script directly:

```bash
# Activate your virtual environment
source venv/bin/activate

# Run the demo
python scripts/casalingua_demo.py
```

Note: This requires the CasaLingua API to be already running in a separate terminal.

## Demo Controls

- The demo will run for approximately 2 minutes
- Press Ctrl+C at any time to exit the demo
- Each demonstration section will run for a few seconds before moving to the next

## Teaching Notes

When using this demo for teaching purposes:

1. **Translation**: Highlight the quality difference between MBART and other translation models
2. **Simplification**: Explain how this helps make complex legal or technical content accessible
3. **Veracity Audit**: Discuss how this ensures translation quality for critical communications
4. **Architecture**: Point out that all processing happens locally (no data sent to external APIs)

## Customization

To modify the demo:

- Edit `casalingua_demo.py` to change the sample sentences or demonstration flow
- Adjust the demo duration (default: 120 seconds) by modifying the `duration` parameter
- Add new demonstration sections by implementing additional methods in the `CasaLinguaDemo` class