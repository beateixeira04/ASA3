import sys
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD


def parse_input():
    lines = sys.stdin.read().strip().splitlines()
    n, m, t = map(int, lines[0].split())

    factories = {}
    countries = {}
    children = []
    children_per_country = {i: 0 for i in range(1, m + 1)}

    # Parse countries first
    for i in range(n + 1, n + m + 1):
        country_id, max_export, min_delivery = map(int, lines[i].split())
        max_export = max(0, max_export)
        min_delivery = max(0, min_delivery)
        countries[country_id] = {
            'max_export': max_export,
            'min_delivery': min_delivery
        }

    # Parse factories
    factories_per_country = {i: [] for i in range(1, m + 1)}
    for i in range(1, n + 1):
        factory_id, country_id, stock = map(int, lines[i].split())
        factories[factory_id] = {'country': country_id, 'stock': stock}
        factories_per_country[country_id].append(factory_id)

    # Parse children
    children_by_country = {}
    factory_to_children = {factory_id: [] for factory_id in factories}
    for i in range(n + m + 1, n + m + t + 1):
        data = list(map(int, lines[i].split()))
        child_id = data[0]
        country_id = data[1]
        
        if country_id not in children_by_country:
            children_by_country[country_id] = []
        children_by_country[country_id].append(child_id)
        
        children_per_country[country_id] += 1
        
        # Add the child to valid factories' list while checking for stock > 0
        valid_requests = []
        for factory_id in data[2:]:
            fac_country_id = factories[factory_id]["country"]
            if factories[factory_id]["stock"] > 0 and (
                    countries[fac_country_id]["max_export"] > 0 or
                    (countries[fac_country_id]["max_export"] == 0 and country_id == fac_country_id)
            ):
                factory_to_children[factory_id].append(child_id)
                valid_requests.append(factory_id)
        
        children.append({"id": child_id, "country": country_id, "factories": valid_requests})

    return (n, t, factories, countries, children, children_per_country,
            factory_to_children, factories_per_country, children_by_country)


def check_min_delivery_feasibility(countries, children_per_country):
    """Check if minimum delivery requirements can potentially be met"""
    for country_id, country in countries.items():
        if country['min_delivery'] > 0:
            if children_per_country[country_id] < country['min_delivery']:
                return False
    return True


def solve_lp():
    n, t, factories, countries, children, children_per_country, factory_to_children, factories_per_country, children_by_country = parse_input()

    if n == 0:
        return -1
    if t == 0:
        return 0

    # Check if minimum delivery requirements are feasible
    if not check_min_delivery_feasibility(countries, children_per_country):
        return -1

    prob = LpProblem("MerryHanukkah", LpMaximize)

    # Decision variables
    x = {}
    for child in children:
        if not child["factories"]:  # If the child has no valid factories, they can't get a present
            continue
        for factory_id in child["factories"]:
            x[f"{child['id']}_{factory_id}"] = LpVariable(
                f"x_{child['id']}_{factory_id}",
                cat="Binary"
            )

    # Objective: Maximize satisfied children
    prob += lpSum(
        lpSum(x.get(f"{child['id']}_{factory_id}", 0)
              for factory_id in child["factories"])
        for child in children if child["factories"]
    )

    # Factory stock constraints
    for factory_id, factory in factories.items():
        prob += lpSum(
            x.get(f"{child_id}_{factory_id}", 0)
            for child_id in factory_to_children[factory_id]
        ) <= factory["stock"]

    # One toy per child
    for child in children:
        prob += lpSum(
            x.get(f"{child['id']}_{factory_id}", 0)
            for factory_id in child["factories"]
        ) <= 1

    # Country constraints - combine export and min delivery
    for country_id, country in countries.items():
        # Count toys exported from this country
        country_exports = lpSum(
            x.get(f"{child}_{factory_id}", 0)
            for factory_id in factories_per_country[country_id]  # Only factories in this country
            for child in factory_to_children[factory_id]  # For each child this factory can serve
            if children[child - 1]['country'] != country_id  # Exports only count for different countries
        )

        prob += country_exports <= country["max_export"]

        # Minimum delivery requirement
        country_deliveries = lpSum(
            lpSum(x.get(f"{child}_{factory_id}", 0)
                        for factory_id in children[child - 1]["factories"])
                        for child in (children_by_country[country_id] if country_id in children_by_country else [])
        )

        if country["min_delivery"] >= 0:
            prob += country_deliveries >= country["min_delivery"]

    solver = PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    if status != 1:
        return -1

    return int(round(prob.objective.value()))


if __name__ == "__main__":
    result = solve_lp()
    print(result)
