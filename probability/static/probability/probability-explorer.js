document.addEventListener(
    "DOMContentLoaded",
    () => {
        const form =
            document.getElementById(
                "probability-explorer-form"
            );

        if (!form) {
            return;
        }

        const config = JSON.parse(
            document.getElementById(
                "probability-ui-config"
            ).textContent
        );

        const state = JSON.parse(
            document.getElementById(
                "probability-explorer-state"
            ).textContent
        );

        const errors = JSON.parse(
            document.getElementById(
                "probability-explorer-errors"
            ).textContent
        );

        const distributionSelect =
            document.getElementById(
                "explorer_distribution"
            );

        const viewSelect =
            document.getElementById(
                "explorer_view"
            );

        const parameterFields =
            document.getElementById(
                "explorer-parameter-fields"
            );

        const noParameters =
            document.getElementById(
                "explorer-no-parameters"
            );

        const parameterization =
            document.getElementById(
                "explorer-parameterization"
            );

        const description =
            document.getElementById(
                "explorer-distribution-description"
            );

        const resetButton =
            document.getElementById(
                "explorer-reset-button"
            );

        let firstRender = true;


        function currentDistribution() {
            return config.distributions[
                distributionSelect.value
            ];
        }


        function explorerViews(spec) {
            if (spec.category === "discrete") {
                return [
                    {
                        key: "pmf",
                        label:
                            "Probability mass (PMF)"
                    },
                    {
                        key: "cdf",
                        label:
                            "Cumulative distribution (CDF)"
                    },
                    {
                        key: "survival",
                        label:
                            "Survival function"
                    }
                ];
            }

            const views = [
                {
                    key: "pdf",
                    label:
                        "Probability density (PDF)"
                },
                {
                    key: "cdf",
                    label:
                        "Cumulative distribution (CDF)"
                },
                {
                    key: "survival",
                    label:
                        "Survival function"
                }
            ];

            if (spec.supports_hazard) {
                views.push({
                    key: "hazard",
                    label:
                        "Hazard function"
                });
            }

            return views;
        }


        function createParameterField(
            parameter,
            value,
            errorMessage
        ) {
            const wrapper =
                document.createElement("div");

            wrapper.className =
                "probability-dynamic-field";

            const label =
                document.createElement("label");

            const inputId =
                `explorer_param_${parameter.name}`;

            label.htmlFor = inputId;

            label.textContent =
                `${parameter.label}`
                + (
                    parameter.symbol
                        ? ` (${parameter.symbol})`
                        : ""
                );

            const input =
                document.createElement("input");

            input.type = "number";
            input.id = inputId;
            input.name = inputId;

            input.className =
                "probability-control";

            input.step =
                parameter.kind === "int"
                    ? "1"
                    : "any";

            if (
                parameter.min_value
                !== null
                && parameter.min_value
                !== undefined
            ) {
                input.min =
                    parameter.min_value;
            }

            if (
                parameter.max_value
                !== null
                && parameter.max_value
                !== undefined
            ) {
                input.max =
                    parameter.max_value;
            }

            input.value = value;

            wrapper.appendChild(label);
            wrapper.appendChild(input);

            if (parameter.help_text) {
                const help =
                    document.createElement(
                        "div"
                    );

                help.className =
                    "probability-field-help";

                help.textContent =
                    parameter.help_text;

                wrapper.appendChild(help);
            }

            if (errorMessage) {
                input.classList.add(
                    "is-invalid"
                );

                const error =
                    document.createElement(
                        "div"
                    );

                error.className =
                    "probability-invalid-feedback";

                error.textContent =
                    errorMessage;

                wrapper.appendChild(error);
            }

            return wrapper;
        }


        function renderParameters(spec) {
            parameterFields.innerHTML = "";

            if (!spec.parameters.length) {
                noParameters.hidden = false;
                return;
            }

            noParameters.hidden = true;

            spec.parameters.forEach(
                (parameter) => {
                    let value =
                        parameter.default;

                    let error = null;

                    if (
                        firstRender
                        && Object
                            .prototype
                            .hasOwnProperty
                            .call(
                                state.parameters,
                                parameter.name
                            )
                    ) {
                        value =
                            state.parameters[
                                parameter.name
                            ];
                    }

                    if (
                        firstRender
                        && errors.parameters
                    ) {
                        error =
                            errors.parameters[
                                parameter.name
                            ];
                    }

                    parameterFields.appendChild(
                        createParameterField(
                            parameter,
                            value,
                            error
                        )
                    );
                }
            );
        }


        function renderViews(spec) {
            const previous =
                firstRender
                    ? state.view
                    : null;

            viewSelect.innerHTML = "";

            const views =
                explorerViews(spec);

            const defaultView =
                spec.category === "continuous"
                    ? "pdf"
                    : "pmf";

            const selected =
                views.some(
                    item =>
                        item.key === previous
                )
                    ? previous
                    : defaultView;

            views.forEach(
                (view) => {
                    const option =
                        document.createElement(
                            "option"
                        );

                    option.value =
                        view.key;

                    option.textContent =
                        view.label;

                    option.selected =
                        view.key === selected;

                    viewSelect.appendChild(
                        option
                    );
                }
            );
        }


        function renderDistribution() {
            const spec =
                currentDistribution();

            if (!spec) {
                return;
            }

            parameterization.textContent =
                spec.parameterization;

            description.textContent =
                spec.description;

            renderParameters(spec);
            renderViews(spec);

            firstRender = false;
        }


        distributionSelect.addEventListener(
            "change",
            () => {
                firstRender = false;
                renderDistribution();
            }
        );


        resetButton.addEventListener(
            "click",
            () => {
                firstRender = false;

                const spec =
                    currentDistribution();

                renderParameters(spec);
                renderViews(spec);
            }
        );


        renderDistribution();
    }
);