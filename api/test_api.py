try:
    from flask_jsonrpc.proxy import ServiceProxy

    server = ServiceProxy("http://localhost:5000/api")

    def test_hello():
        resp = server.App.hello()
        assert resp["result"] == "Hello!"

    def test_create_equilibrium():
        params = {
            "compounds": ["NaCl"],
            "closingEqType": 3,
            "allowPrecipitation": False,
            "initial_feed_mass_balance": [],
        }
        resp = server.App.create_equilibrium(**params)
        assert isinstance(resp["result"]["reactions"], list)

    def test_solve_equilibrium():
        resp = server.App.solve_equilibrium()
        assert isinstance(resp["result"]["reactions"], list)


except ModuleNotFoundError:
    print(
        """Minimal version does not include the API, install the following PYPI packages if you need this functionality:
    [flask, flas_jsonrpc]
    """
    )
# from pyequion.reac


if __name__ == "__main__":
    test_create_equilibrium()
