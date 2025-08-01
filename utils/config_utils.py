def load_best_config():
    """
    Parses a result file and returns a dict of selected hyperparameters.
    """
                                                        
    # Add fall back values here
    
    config = {}

    file_path = os.path.join(os.path.dirname(__file__), "..", "tuning", "bayesian_tuning_results.txt")

    with open(file_path, 'r') as f:
        content = f.read()

    for line in content.splitlines():
        match = re.match(r'(\w+):\s+(.*)', line)
        if match:
            key, value = match.groups()
            if key in selected_keys:
                # Convert values to appropriate types
                if value.lower() in {"true", "false"}:
                    config[key] = value.lower() == "true"
                elif value.replace('.', '', 1).isdigit():
                    if '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                elif value.startswith('"') or value.startswith("'"):
                    config[key] = value.strip('"').strip("'")
                else:
                    try:
                        config[key] = eval(value)
                    except:
                        config[key] = value
    return config