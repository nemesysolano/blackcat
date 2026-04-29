CREATE OR REPLACE FUNCTION "K" (z DOUBLE PRECISION)
returns DOUBLE PRECISION
LANGUAGE plpgsql AS $$
BEGIN
	RETURN 1 + SIN(z) - COS(z);
END;
$$;

CREATE OR REPLACE FUNCTION "Q" (z DOUBLE PRECISION)
returns DOUBLE PRECISION
LANGUAGE plpgsql AS $$
BEGIN
	RETURN z + POWER(SIN(Z),2);
END;
$$;

CREATE OR REPLACE FUNCTION "FH" ("ϕ1" DOUBLE PRECISION, "ϕ2" DOUBLE PRECISION)
returns DOUBLE PRECISION
LANGUAGE plpgsql AS $$
BEGIN
	RETURN ((16/(power(pi(),2) + 2*pi() + 16))/8)*("ϕ2" * "Q"("ϕ1") + "ϕ1"*"Q"("ϕ2") + 2 * "K"("ϕ1")*"K"("ϕ2"));
END;
$$;


CREATE OR REPLACE FUNCTION "FΩ"("θ1" double precision, "θ2" double precision, "θ3" double precision, "θ4" double precision)
RETURNS double precision
LANGUAGE plpgsql AS $$
BEGIN
    RETURN ((128/(power(pi(),4) + 2*power(pi(),3) + 48*power(pi(),2)))/32) * (
        -- 4 Squared Terms: Each variable evaluated with Q(), multiplied by the remaining 3 angles
        "Q"("θ1") * "θ2" * "θ3" * "θ4" +
        "Q"("θ2") * "θ1" * "θ3" * "θ4" +
        "Q"("θ3") * "θ1" * "θ2" * "θ4" +
        "Q"("θ4") * "θ1" * "θ2" * "θ3" +
        
        -- 6 Cross-Product Terms: Every unique pair evaluated with C(), multiplied by the remaining 2 angles
        2 * (
            "K"("θ1") * "K"("θ2") * "θ3" * "θ4" +
            "K"("θ1") * "K"("θ3") * "θ2" * "θ4" +
            "K"("θ1") * "K"("θ4") * "θ2" * "θ3" +
            "K"("θ2") * "K"("θ3") * "θ1" * "θ4" +
            "K"("θ2") * "K"("θ4") * "θ1" * "θ3" +
            "K"("θ3") * "K"("θ4") * "θ1" * "θ2"
        )
    );
END;
$$;

CREATE OR REPLACE FUNCTION "F" ("θ1" double precision, "θ2" double precision, "θ3" double precision, "θ4" double precision, "ϕ1" DOUBLE PRECISION, "ϕ2" DOUBLE PRECISION)
returns DOUBLE PRECISION
LANGUAGE plpgsql AS $$
BEGIN
	RETURN ("FΩ"("θ1", "θ2", "θ3", "θ4") + "FH"("ϕ1", "ϕ2"))/2;
END;
$$;
