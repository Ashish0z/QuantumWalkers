import numpy as np

def generate_realistic_sonar_data(n_samples=1000, seed=42):
    """Generate more realistic sonar data with controlled complexity"""
    np.random.seed(seed)
    X, y = [], []
    
    for _ in range(n_samples):
        cls = np.random.choice([0, 1, 2])
        t = np.linspace(0, 1, 60)  # Reduced from 100 to make features more manageable
        
        if cls == 0:  # Submarine
            # Simple frequency signature with harmonics
            fundamental = 150 + np.random.normal(0, 10)
            sig = np.sin(2*np.pi*fundamental*t) + 0.3*np.sin(2*np.pi*2*fundamental*t)
            
        elif cls == 1:  # Whale
            # Frequency sweep (whale song characteristic)
            f0, f1 = 80 + np.random.normal(0, 5), 200 + np.random.normal(0, 10)
            sig = np.sin(2*np.pi*(f0 + (f1-f0)*t)*t)
            
        else:  # Debris
            # Broadband noise with some structure
            sig = np.random.normal(0, 0.5, len(t))
            # Add some periodic component
            sig += 0.4*np.sin(2*np.pi*(300 + np.random.normal(0, 20))*t)
            
        # Add realistic noise
        noise = np.random.normal(0, 0.2, len(sig))
        X.append(sig + noise)
        y.append(cls)
    
    return np.array(X), np.array(y)