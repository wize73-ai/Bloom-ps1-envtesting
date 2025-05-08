#!/usr/bin/env python3
"""
CasaLingua Short Demo

A shortened version of the complete demo that runs for only 45 seconds,
showing all major functionality.

This demo showcases:
- Translation capabilities with metrics
- Text simplification for housing documents
- Cost analysis for Bloom Housing integration (1.8M annual hits)

Cost calculation summary:
- Infrastructure: ~$256/month (AWS Medium Deployment)
- ML Processing: ~$264/month (based on token volume)
- Total Cost: ~$520/month ($6,240 annually)
- ROI: ~7,400% ($708,100 annual savings vs $6,240 cost)
- Payback period: <1 month

Usage:
    python casalingua_short_demo.py
"""

import os
import sys
import asyncio

# Import the main demo
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from scripts.casalingua_complete_demo import CasaLinguaDemo

async def main():
    """Main function to run the shortened demo"""
    try:
        # Create a demo with 45 seconds duration, connecting to running CasaLingua
        demo = CasaLinguaDemo(duration=45, simulate=False)  
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())