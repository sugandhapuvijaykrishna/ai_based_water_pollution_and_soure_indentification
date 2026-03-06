# Krishna River Pollution Intelligence System

> AI-powered satellite and sensor fusion platform for
> real-time water pollution detection on Krishna River,
> Vijayawada, Andhra Pradesh.
> Built for Innovathon 2026 — Green Energy and Sustainable Development

---

## What This Project Does

This system detects and maps water pollution across 30 kilometers
of Krishna River using:
- Real Sentinel-2 satellite imagery (7 seasonal scenes)
- Physics-based IoT sensor simulation (5 stations, exponential decay model)
- Dual-Branch CNN + MLP deep learning model
- Interactive Streamlit dashboard with ArcGIS satellite map

Output: 2,764 geo-tagged pollution predictions with NTU values,
risk levels, flow direction, and health index — visualized on
a live satellite map with click interactions and downloadable AI report.

Addresses UN SDG Goal 6 — Clean Water and Sanitation.

---

## Folder Structure
Main/
├── README.md              ← This file
├── requirements.txt       ← Python dependencies
├── DEMO_GUIDE.md         ← Demo day checklist
├── .gitignore            ← Git ignore rules
│
├── pipeline/             ← All data and model scripts
│   ├── generate_sensor_data.py    ← Creates synthetic sensor CSVs
│   ├── generate_synthetic_data.py ← Creates polluted TIFFs from raw bands
│   ├── fusion_dataset.py          ← Builds NPZ from TIFFs + sensors
│   ├── build_river_dataset.py     ← Filters river-only patches via NDWI
│   ├── trainer_fusion.py          ← Trains dual-branch CNN model
│   ├── trainer_river.py           ← Trains binary river detector
│   ├── verify_integration.py      ← Checks all pipeline outputs
│   ├── data_audit.py              ← Full data quality audit
│   ├── check_gpu.py               ← GPU availability check
│   ├── final_audit.py             ← Legacy audit script
│   └── verify_npz.py              ← NPZ file inspector
│
├── dashboard/
│   └── command_center.py          ← Streamlit dashboard (3 tabs)
│
├── model_core/           ← Core ML modules
│   ├── preprocessing.py           ← Sentinel-2 band processing
│   ├── patch_extraction.py        ← 32x32 patch extractor with NDWI mask
│   ├── model.py                   ← CNN and DualBranchFusionModel definitions
│   ├── data_pipeline.py           ← Train/val split and NPZ loader
│   ├── training.py                ← Training utilities and callbacks
│   ├── evaluation.py              ← Classification metrics
│   ├── evaluation_regression.py   ← Regression metrics
│   ├── export_geotiff.py          ← GeoTIFF prediction export
│   ├── gpu_setup.py               ← GPU memory configuration
│   └── main_pipeline.py           ← Full Landsat classification pipeline
│
├── satellite_data/       ← Raw Sentinel-2 TIFF bands
│   ├── April/            ← 2025-04-01 scene (B02, B03, B04, B08)
│   ├── May/              ← 2025-05-11 scene
│   ├── Jul/              ← 2025-07-08 scene
│   ├── Sept/             ← 2025-09-06 scene
│   ├── Nov/              ← 2025-11-07 scene
│   ├── Nov (2)/          ← 2025-11-15 scene
│   └── Dec/              ← 2025-12-15 scene
│
├── outputs/              ← All generated files (auto-created)
│   ├── *_sensors.csv              ← Per-month sensor readings
│   ├── ground_truth_sensors.csv   ← Combined sensor dataset
│   ├── *_polluted.tif             ← Synthetic polluted scenes
│   ├── *_labels.tif               ← Pollution label rasters
│   ├── final_fusion_dataset.npz   ← River-only training patches
│   ├── river_dataset.npz          ← Binary river detection dataset
│   ├── best_fusion_model.h5       ← Trained NTU regression model
│   ├── river_model.h5             ← Trained river detector model
│   ├── results_for_viz.csv        ← Final dashboard data
│   └── river_metrics.txt          ← River model evaluation metrics
│
└── archive/              ← Old logs and one-time scripts

---

## How It Works — Full Pipeline
Step 1: Raw Sentinel-2 TIFFs (B03, B04, B08)
↓
Step 2: generate_synthetic_data.py
→ Computes NDWI mask
→ Injects Gaussian pollution plume
→ Outputs: outputs/*_polluted.tif + _labels.tif
↓
Step 3: generate_sensor_data.py
→ 5 virtual stations along river
→ Exponential decay NTU model
→ Seasonal calibration (CPCB ranges)
→ Outputs: outputs/_sensors.csv
↓
Step 4: fusion_dataset.py
→ NDWI water masking (threshold > 0.10)
→ 32x32 patch extraction (stride 16)
→ Nearest sensor NTU assignment
→ Outputs: outputs/final_fusion_dataset.npz
↓
Step 5: trainer_fusion.py
→ Dual-Branch CNN + MLP training
→ 50 epochs, Early Stopping, ReduceLROnPlateau
→ KNN flow vector calculation
→ Outputs: best_fusion_model.h5 + results_for_viz.csv
↓
Step 6: streamlit run dashboard/command_center.py
→ Reads results_for_viz.csv
→ Renders satellite map + charts + AI report

---

## Setup Instructions

### Prerequisites
Python     : 3.9 or above
RAM        : 4GB minimum, 8GB recommended
GPU        : Optional but speeds up training 3-5x
Disk space : 2GB for all outputs

### Installation
```bash
cd Main
pip install -r requirements.txt
```

### Satellite Data Download

Go to https://apps.sentinel-hub.com/eo-browser
Search location: Vijayawada, Andhra Pradesh
Select: Sentinel-2 L2A
Download bands: B03, B04, B08 for each date
Dates used: 2025-04-01, 2025-05-11, 2025-07-08,
2025-09-06, 2025-11-07, 2025-11-15, 2025-12-15
Save each date in satellite_data/[Month]/ folder


## Run Order — Step by Step
```bash
# Step 1 — Generate sensor CSVs
python pipeline/generate_sensor_data.py

# Step 2 — Generate polluted satellite scenes
python pipeline/generate_synthetic_data.py

# Step 3 — Build fusion dataset (river patches only)
python pipeline/fusion_dataset.py

# Step 4 — Build river-only binary dataset
python pipeline/build_river_dataset.py

# Step 5 — Train NTU regression model
python pipeline/trainer_fusion.py

# Step 6 — Verify everything is correct
python pipeline/data_audit.py

# Step 7 — Launch dashboard
streamlit run dashboard/command_center.py
```
Dashboard opens at: http://localhost:8501

## Dashboard — Tab by Tab
### Tab 1 — Executive Dashboard
| Element | Description |
| :--- | :--- |
| Avg Turbidity | Mean NTU across all river zones |
| Peak Turbidity | Maximum NTU detected |
| Critical Zones | Count of zones above 500 NTU |
| High Risk Zones | Count of zones 300-500 NTU |
| River Health Index | Score 0-100 (formula: 100 - NTU/8) |
| Pie chart | Risk level distribution |
| Histogram | NTU distribution with threshold lines |
| Scatter plot | Spatial pollution map (Lat vs Lon) |
| Table | Top 20 highest NTU readings |

### Tab 2 — Pollution Map
| Element | Description |
| :--- | :--- |
| Base map | Real ArcGIS satellite imagery — no API key |
| Colored dots | Green=Low, Yellow=Moderate, Orange=High, Red=Critical |
| Click a dot | Shows blue flow direction arrow on map |
| White ring | Highlights the selected zone |
| Detail panel | NTU, risk, health, source, cardinal flow direction |
| Sidebar | Filter by risk level, NTU threshold, max points |

### Tab 3 — AI Report
| Element | Description |
| :--- | :--- |
| Executive summary | Overall river status and key numbers |
| Zone breakdown | Count and % per risk level |
| Source analysis | Industrial vs Agricultural attribution |
| Action plan | Immediate / Short-term / Long-term recommendations |
| Download button | Exports full report as .txt file |
| Source breakdown | Industrial and Agricultural cards with stats |
| Priority zones | Top 4 intervention coordinates |
| Model performance | Architecture, patches, coverage summary |

## Data Description
### Satellite Input
| Property | Value |
| :--- | :--- |
| Sensor | Sentinel-2 L2A |
| Resolution | 10 meters per pixel |
| Bands used | B03 Green, B04 Red, B08 NIR |
| Scenes | 7 dates across 3 seasons |
| Coverage | Krishna River, 80.43E to 80.70E |
| Source | ESA Copernicus Open Access Hub |

### Sensor Simulation
| Property | Value |
| :--- | :--- |
| Stations | 5 (KR-01 to KR-05) |
| Method | Exponential decay: NTU = base × e^(-k × distance) |
| Decay constant | k = 0.08 |
| Summer NTU | 400-800 (peak industrial discharge) |
| Monsoon NTU | 100-300 (dilution + agricultural runoff) |
| Winter NTU | 10-50 (baseline clean water) |
| Calibration | CPCB published seasonal ranges |

### Model Output
| Property | Value |
| :--- | :--- |
| Total patches | 2,764 (after outlier filtering) |
| NTU range | 0 to 837 |
| NTU mean | 166.7 |
| Flow coverage | 100% (all dots have direction) |
| Health index | 79.2% |
| Coordinate range | 16.49-16.58N, 80.44-80.68E |

## Model Architecture
### Branch 1 — CNN (Satellite Patches)
Input: (32, 32, 6)
→ Conv2D(32, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.3)
→ Conv2D(64, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.3)
→ Conv2D(128, 3x3) → BatchNorm → ReLU → GlobalAvgPool → Dropout(0.3)
→ Dense(64, ReLU)

### Branch 2 — MLP (Geographic Location)
Input: (2,) → [Latitude, Longitude]
→ Dense(16, ReLU)
→ Dense(8, ReLU)

### Fusion + Output
Concatenate [Branch1(64), Branch2(8)] → (72,)
→ Dense(32, ReLU)
→ Dense(1, Linear) → NTU prediction

### Training Configuration
Loss function  : Mean Squared Error
Metric         : Mean Absolute Error
Optimizer      : Adam (lr=0.001)
Epochs         : 50 maximum
Early stopping : patience=15
LR scheduler   : ReduceLROnPlateau (factor=0.5, patience=7)
Batch size     : 64
Train/val split: 80% / 20%

## NTU Risk Level Reference
| NTU Range | Risk Level | Color | Meaning |
| :--- | :--- | :--- | :--- |
| 0 — 100 | Low | Green | Safe for all uses |
| 100 — 300 | Moderate | Yellow | Caution — treatment needed |
| 300 — 500 | High | Orange | Restricted use only |
| 500+ | Critical | Red | Dangerous — immediate action |

WHO safe drinking limit: 5 NTU
Our critical zones peak: 837 NTU (167x above safe limit)

## Key Files Quick Reference
| File | What It Does |
| :--- | :--- |
| pipeline/fusion_dataset.py | Extracts river patches from TIFFs |
| pipeline/trainer_fusion.py | Trains the NTU regression model |
| pipeline/data_audit.py | Verifies all data before demo |
| dashboard/command_center.py | Streamlit dashboard — main entry point |
| model_core/patch_extraction.py | 32x32 patch extractor with NDWI mask |
| model_core/model.py | DualBranchFusionModel architecture |
| model_core/preprocessing.py | Sentinel-2 band processing |
| outputs/results_for_viz.csv | Final predictions read by dashboard |
| outputs/best_fusion_model.h5 | Trained model weights |

## Known Limitations
- Sensor NTU values are physics-based simulation — not real IoT measurements
- Model trained on 7 scenes — more seasonal scenes improve accuracy
- Flow direction is KNN spatial gradient approximation — not actual hydrological data
- Dashboard requires local Streamlit server — not yet cloud deployed
- Summer NTU simulation underestimates peak values at far stations

## Future Roadmap
| Priority | Feature | Timeline |
| :--- | :--- | :--- |
| 1 | Google Earth Engine API for daily auto-fetch | Week 1 post-demo |
| 2 | Real CPCB sensor data integration | Week 2 |
| 3 | SMS/WhatsApp alerts for critical zones | Week 3 |
| 4 | Google Cloud Run deployment | Week 4 |
| 5 | Extend to Godavari and Ganga rivers | Month 2 |

## Tech Stack
| Component | Technology | Version |
| :--- | :--- | :--- |
| Satellite Processing | Rasterio | >=1.3.0 |
| Deep Learning | TensorFlow + Keras | >=2.12.0 |
| Spatial Analysis | Scikit-learn KNN | >=1.2.0 |
| Numerical Computing | NumPy + SciPy | >=1.23.0 |
| Data Processing | Pandas | >=1.5.0 |
| Dashboard | Streamlit | >=1.28.0 |
| Visualization | Plotly | >=5.14.0 |
| Map Tiles | ArcGIS World Imagery | Free/Public |

## Team
[Team Name] — [College Name]
Innovathon 2026 — Green Energy and Sustainable Development

## License
Developed for educational and research purposes
as part of Innovathon 2026 competition.
Not for commercial use.
