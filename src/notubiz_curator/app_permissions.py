"""Bootstrap permissions for a generated Databricks App identity."""

from urllib.parse import quote

from databricks.sdk import WorkspaceClient


def grant_app_secret_access(workspace: WorkspaceClient, catalog: str, schema: str, principal: str) -> None:
    """Grant only parent usage and read access to the two SPN secrets."""
    grants = [
        ("catalog", catalog, "USE_CATALOG"),
        ("schema", f"{catalog}.{schema}", "USE_SCHEMA"),
        ("secret", f"{catalog}.{schema}.databricks_client_id", "READ_SECRET"),
        ("secret", f"{catalog}.{schema}.databricks_client_secret", "READ_SECRET"),
    ]
    for securable_type, full_name, privilege in grants:
        # READ_SECRET is not yet in the SDK's Privilege enum. Use its API client
        # because SQL secret grants are unavailable on the course compute.
        workspace.api_client.do(
            "PATCH",
            f"/api/2.1/unity-catalog/permissions/{securable_type}/{quote(full_name, safe='')}",
            body={"changes": [{"principal": principal, "add": [privilege]}]},
        )


def grant_runtime_trace_access(workspace: WorkspaceClient, catalog: str, schema: str, principal: str) -> None:
    """Grant trace access to the runtime SPN using the supplied grantor client."""
    from databricks.sdk.service.catalog import PermissionsChange, Privilege, TableType

    for table in workspace.tables.list(catalog_name=catalog, schema_name=schema):
        if table.name and table.name.startswith("llmops_"):
            privileges = [Privilege.SELECT]
            if table.table_type != TableType.VIEW:
                privileges.append(Privilege.MODIFY)
            workspace.grants.update(
                securable_type="table",
                full_name=f"{catalog}.{schema}.{table.name}",
                changes=[PermissionsChange(principal=principal, add=privileges)],
            )
