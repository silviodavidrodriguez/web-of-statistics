document.addEventListener("DOMContentLoaded", () => {
    const configElement = document.getElementById(
        "probability-ui-config"
    );

    const stateElement = document.getElementById(
        "probability-form-state"
    );

    const errorsElement = document.getElementById(
        "probability-field-errors"
    );

    const form = document.getElementById(
        "probability-functions-form"
    );

    if (
        !configElement
        || !stateElement
        || !errorsElement
        || !form
    ) {
        return;
    }

    const config = JSON.parse(
        configElement.textContent
    );

    const initialState = JSON.parse(
        stateElement.textContent
    );

    const initialErrors = JSON.parse(
        errorsElement.textContent
    );

    const distributionSelect =
        document.getElementById("distribution");

    const operationSelect =
        document.getElementById("operation");

    const parameterFields =
        document.getElementById(
            "probability-parameter-fields"
        );

    const inputFields =
        document.getElementById(
            "probability-input-fields"
        );

    const noParameters =
        document.getElementById(
            "probability-no-parameters"
        );

    const parameterization =
        document.getElementById(
            "probability-parameterization"
        );

    const distributionDescription =
        document.getElementById(
            "probability-distribution-description"
        );

    const operationDescription =
        document.getElementById(
            "probability-operation-description"
        );

    const resetButton =
        document.getElementById(
            "probability-reset-button"
        );

    let firstRender = true;


    function currentDistribution() {
        return config.distributions[
            distributionSelect.value
        ];
    }


    function operationRegistry(spec) {
        return config.operations[
            spec.category
        ];
    }


    function replaceSymbol(text, symbol) {
        return text.replaceAll(
            "{symbol}",
            symbol
        );
    }


    function createField(
        definition,
        prefix,
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
            `${prefix}_${definition.name}`;

        label.htmlFor = inputId;

        let labelText = definition.label;

        if (definition.symbol) {
            labelText +=
                ` (${definition.symbol})`;
        }

        label.textContent = labelText;

        const input =
            document.createElement("input");

        input.type = "number";
        input.id = inputId;
        input.name = inputId;

        input.className =
            "probability-control";

        input.step =
            definition.kind === "int"
                ? "1"
                : "any";

        if (
            definition.min_value
            !== null
            && definition.min_value
            !== undefined
        ) {
            input.min =
                definition.min_value;
        }

        if (
            definition.max_value
            !== null
            && definition.max_value
            !== undefined
        ) {
            input.max =
                definition.max_value;
        }

        if (
            value !== null
            && value !== undefined
        ) {
            input.value = value;
        }

        wrapper.appendChild(label);
        wrapper.appendChild(input);

        if (definition.help_text) {
            const help =
                document.createElement("div");

            help.className =
                "probability-field-help";

            help.textContent =
                definition.help_text;

            wrapper.appendChild(help);
        }

        if (errorMessage) {
            input.classList.add(
                "is-invalid"
            );

            const error =
                document.createElement("div");

            error.className =
                "probability-invalid-feedback";

            error.textContent =
                errorMessage;

            wrapper.appendChild(error);
        }

        return wrapper;
    }


    function parameterValue(
        parameter
    ) {
        if (
            firstRender
            && Object.prototype.hasOwnProperty.call(
                initialState.parameters,
                parameter.name
            )
        ) {
            return initialState.parameters[
                parameter.name
            ];
        }

        return parameter.default;
    }


    function inputValue(
        definition
    ) {
        if (
            firstRender
            && Object.prototype.hasOwnProperty.call(
                initialState.inputs,
                definition.name
            )
        ) {
            return initialState.inputs[
                definition.name
            ];
        }

        return definition.default;
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
                const error =
                    firstRender
                        ? initialErrors.parameters[
                            parameter.name
                        ]
                        : null;

                parameterFields.appendChild(
                    createField(
                        parameter,
                        "param",
                        parameterValue(
                            parameter
                        ),
                        error
                    )
                );
            }
        );
    }


    function renderOperationOptions(
        spec
    ) {
        operationSelect.innerHTML = "";

        const registry =
            operationRegistry(spec);

        let selectedOperation =
            spec.default_operation;

        if (
            firstRender
            && spec.operations.includes(
                initialState.operation
            )
        ) {
            selectedOperation =
                initialState.operation;
        }

        spec.operations.forEach(
            (operationKey) => {
                const operation =
                    registry[operationKey];

                const option =
                    document.createElement(
                        "option"
                    );

                option.value =
                    operationKey;

                option.textContent =
                    `${operation.label} — `
                    + replaceSymbol(
                        operation.expression,
                        spec.variable_symbol
                    );

                if (
                    operationKey
                    === selectedOperation
                ) {
                    option.selected = true;
                }

                operationSelect.appendChild(
                    option
                );
            }
        );
    }


    function renderOperationInputs(
        spec
    ) {
        const registry =
            operationRegistry(spec);

        const operation =
            registry[
                operationSelect.value
            ];

        operationDescription.textContent =
            operation.description;

        inputFields.innerHTML = "";

        operation.inputs.forEach(
            (definition) => {
                const error =
                    firstRender
                        ? initialErrors.inputs[
                            definition.name
                        ]
                        : null;

                inputFields.appendChild(
                    createField(
                        definition,
                        "input",
                        inputValue(
                            definition
                        ),
                        error
                    )
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

        distributionDescription.textContent =
            spec.description;

        renderParameters(spec);
        renderOperationOptions(spec);
        renderOperationInputs(spec);

        firstRender = false;
    }


    distributionSelect.addEventListener(
        "change",
        () => {
            firstRender = false;
            renderDistribution();
        }
    );


    operationSelect.addEventListener(
        "change",
        () => {
            firstRender = false;

            renderOperationInputs(
                currentDistribution()
            );
        }
    );


    resetButton.addEventListener(
        "click",
        () => {
            const spec =
                currentDistribution();

            firstRender = false;

            renderParameters(spec);

            operationSelect.value =
                spec.default_operation;

            renderOperationInputs(spec);
        }
    );


    renderDistribution();
});