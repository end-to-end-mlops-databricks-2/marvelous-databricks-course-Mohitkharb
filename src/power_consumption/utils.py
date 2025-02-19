import re 

def get_uc_table_name(catalog_name: str, schema_name: str, use_case_name: str, type: str, branch_name: str, run_id: str) -> str:
    """
    Generate a unique table name for a Unity Catalog table based on the given parameters.

    Args:
        catalog_name (str): The name of the Unity Catalog catalog.
        schema_name (str): The name of the Unity Catalog schema.
        use_case_name (str): The name of the use case.
        type (str): The type of the table.
        branch_name (str): The name of the branch.
        run_id (str): The run id.
    """

    if branch_name:
        match = re.search(r"[^/]+$", branch_name)
        branch_name = match.group() if match else ""
    
    return f"{catalog_name}.{schema_name}.{use_case_name}_{type}_{branch_name}"

def get_experiment_name(workspace_location: str, user_name: str, type: str, use_case_name: str, branch_name: str) -> str:
    """
    Generate a unique experiment name based on the given parameters.

    Args:
        workspace_location (str): The location of the workspace. Workspace or projects
        user_name (str): The name of the user or the databricks group
        type (str): The type of the model.
        use_case_name (str): The name of the use case.
        branch_name (str): The name of the branch.
        run_id (str): The run id.
    """
    if branch_name:
        match = re.search(r"[^/]+$", branch_name)
        branch_name = match.group() if match else ""


    return f"/{workspace_location}/{user_name}/{use_case_name}_{type}_{branch_name}"

def get_model_name(use_case_name: str, type: str, branch_name: str,) -> str:
    """
    Generate a unique model name based on the given parameters.
    """
    if branch_name:
        match = re.search(r"[^/]+$", branch_name)
        branch_name = match.group() if match else ""

    return f"{use_case_name}_{type}_{branch_name}"

