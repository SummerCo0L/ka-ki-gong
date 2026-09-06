# Social Service Office (SSO) Contact List

Source: "Social Service Office Contact List", Social Support Group, Ministry of
Social and Family Development (MSF), updated as of August 2026. See
`sso_locations.json` in this folder for the structured data used by
`retrieval/sso_lookup.py` and the `find_nearest_sso` tool
(`processors/sso_tool.py`).

ComCare Hotline: 1800-222-0000 (daily, 7am to midnight) — call this number if
you are unsure which SSO is nearest to you.

| # | SSO | Address |
|---|-----|---------|
| 1 | Social Service Office @ Ang Mo Kio | ServiceSG Centre Ang Mo Kio, 53 Ang Mo Kio Avenue 3, AMK Hub #04-02, Singapore 569933 |
| 2 | Social Service Office @ Bedok | 21 Bedok North St 1 #01-02, Singapore 469659 |
| 3 | Social Service Office @ Boon Lay | NKF Integrated Renal Centre, 500 Corporation Road #01-01, Singapore 649808 |
| 4 | Social Service Office @ Bukit Batok | Blk 369 Bukit Batok St 31 #01-505, Singapore 650369 |
| 5 | Social Service Office @ Bukit Merah | Bukit Merah Community Hub, 3779 Jalan Bukit Merah #01-01, Singapore 159462 |
| 6 | Social Service Office @ Bukit Panjang | Blk 232 Pending Road #01-29, Singapore 670232 |
| 7 | Social Service Office @ Clementi | Blk 358 Clementi Ave 2 #01-285, Singapore 120358 |
| 8 | Social Service Office @ Chua Chu Kang | 8A Teck Whye Lane, Singapore 681008 |
| 9 | Social Service Office @ Geylang Serai | 10 Eunos Road 8, #12-02 Singapore Post Centre, Singapore 408600 |
| 10 | Social Service Office @ Hougang | Blk 662 Hougang Ave 4 #01-413, Singapore 530662 |
| 11 | Social Service Office @ Jalan Besar | Jalan Besar Community Club, 69 Jellicoe Road, #01-03 Singapore 208737 |
| 12 | Social Service Office @ Jurong East | Devan Nair Institute for Employment and Employability, 80 Jurong East Street 21, #01-07 Singapore 609607 |
| 13 | Social Service Office @ Kreta Ayer | Kreta Ayer Community Club, 28A Kreta Ayer Road, #01-03 Singapore 088995 |
| 14 | Social Service Office @ Pasir Ris | 120 Pasir Ris Central #01-16, Pasir Ris Sports Centre, Singapore 519640 |
| 15 | Social Service Office @ Punggol | ServiceSG Centre One Punggol, 1 Punggol Drive, #01-01, Singapore 828629 |
| 16 | Social Service Office @ Queenstown | Block 40 Margaret Drive #02-01, Singapore 140040 |
| 17 | Social Service Office @ Sembawang | Remote only from 24 Aug 2026 — see note below |
| 18 | Social Service Office @ Sengkang | Blk 261C Sengkang East Way #01-506, Singapore 543261 |
| 19 | Social Service Office @ Serangoon | Blk 332 Serangoon Ave 3 #01-257, Singapore 550332 |
| 20 | Social Service Office @ Taman Jurong | 301A Corporation Drive, Singapore 619773 |
| 21 | Social Service Office @ Tampines | ServiceSG Centre, Our Tampines Hub, 1 Tampines Walk #01-21, Singapore 528523 |
| 22 | Social Service Office @ Toa Payoh | 490 Lor 6 Toa Payoh #07-11 HDB Hub Bizthree, Singapore 310490 |
| 23 | Social Service Office @ Woodlands | ServiceSG Centre, 900 South Woodlands Drive, Woodlands Civic Centre #03-01A, Singapore 730900 |
| 24 | Social Service Office @ Yishun | Blk 746 Yishun Street 72 #01-127, Singapore 760746 |

## Sembawang (special arrangement)

From 24 Aug 2026, SSO @ Sembawang operates remotely with no walk-in service.

- Apply for ComCare assistance online via SupportGoWhere, or call the ComCare
  Hotline 1800-222-0000 (daily, 7am to midnight).
- For help resetting SingPass for an online application, visit the nearest
  ServiceSG Centre at Bukit Canberra.
- For other SSO queries, email Ask_SSO@msf.gov.sg.

## Notes on this data

- This list is not embedded into the general scheme retrieval index
  (`data/embeddings.npy`) — it is loaded directly by `retrieval/sso_lookup.py`
  for exact/near-match lookup by area name, since matching "which SSO is
  nearest" is a location-matching problem, not a semantic search problem.
- The 24 SSOs above do not individually cover every Singapore town or
  neighbourhood. `retrieval/sso_lookup.py` maps common nearby
  neighbourhood/MRT names to the geographically closest SSO on this list as a
  best-effort approximation — it is not an official MSF catchment-area
  mapping. Callers should always be given the option to confirm via the
  ComCare Hotline (1800-222-0000) if there is any doubt.
