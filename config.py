from darts.datasets import SunspotsDataset, ETTh1Dataset, EnergyDataset, TaxiNewYorkDataset

DATASETS_CONFIG = {

    "ETTh1": {
        "loader": ETTh1Dataset,
        "target": "OT",
        "horizon": 24,
        "epsilons": [0.1,0.4,0.7],
        "m_multiplier": 2,
        "freq": "h"
    },

    
    "Energy": {
        "loader": EnergyDataset,
        "target": "generation hydro run-of-river and poundage",
        "horizon": 24,
        "epsilons": [5,10,20],
        "m_multiplier": 2,
        "freq": "h"
    },
    "TaxiNewYork": {
        "loader": TaxiNewYorkDataset,
        "target": '#Passengers',
        "horizon": 48,
        "epsilons": [250, 500, 750],
        "m_multiplier": 2,
        "freq": "h"
    },
    "Sunspots": {
        "loader": SunspotsDataset,
        "target": "Sunspots",
        "horizon": 12,
        "epsilons": [5,10,20],
        "m_multiplier": 2,
        "freq": "m"
    }
}
