import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from orchestrator.ocm_client import build_ocm_request_params, fetch_all_stations_paginated

async def main():
    params = build_ocm_request_params(
        lat=20.0063,
        lon=73.7900,
        radius_km=100.0,
        country_code="IN"
    )
    # temporarily disable status and other filters to see what we get
    params["statustypeid"] = None
    params["verbose"] = "false"
    
    print("Fetching from OCM with params:", params)
    data = await fetch_all_stations_paginated(params, max_pages=1)
    print(f"Raw data size: {len(data)}")
    for d in data[:5]:
        addr = d.get('AddressInfo', {})
        print(d.get('ID'), addr.get('Title'), addr.get('Latitude'), addr.get('Longitude'), d.get('StatusTypeID'))

if __name__ == "__main__":
    asyncio.run(main())    
