document.addEventListener(
    "DOMContentLoaded",
    () => {

        const configElement =
            document.getElementById(
                "probability-ui-config"
            );

        if (!configElement) {
            return;
        }

        const config = JSON.parse(
            configElement.textContent
        );


        function initializeLab(options) {

            const form =
                document.getElementById(
                    options.formId
                );

            if (!form) {
                return;
            }


            const state = JSON.parse(
                document.getElementById(
                    options.stateId
                ).textContent
            );

            const errors = JSON.parse(
                document.getElementById(
                    options.errorsId
                ).textContent
            );


            const distributionSelect =
                document.getElementById(
                    options.distributionId
                );

            const parameterFields =
                document.getElementById(
                    options.parameterFieldsId
                );

            const noParameters =
                document.getElementById(
                    options.noParametersId
                );

            const parameterization =
                document.getElementById(
                    options.parameterizationId
                );

            const description =
                document.getElementById(
                    options.descriptionId
                );

            const seed =
                document.getElementById(
                    options.seedId
                );

            const resetButton =
                document.getElementById(
                    options.resetButtonId
                );

            const generateAgain =
                document.getElementById(
                    options.generateAgainId
                );


            let firstRender = true;


            function currentDistribution() {
                return config.distributions[
                    distributionSelect.value
                ];
            }


            function createParameterField(
                parameter,
                value,
                errorMessage
            ) {
                const wrapper =
                    document.createElement(
                        "div"
                    );

                wrapper.className =
                    "probability-dynamic-field";


                const inputId =
                    options.parameterPrefix
                    + parameter.name;


                const label =
                    document.createElement(
                        "label"
                    );

                label.htmlFor = inputId;

                label.textContent =
                    parameter.label
                    + (
                        parameter.symbol
                            ? ` (${parameter.symbol})`
                            : ""
                    );


                const input =
                    document.createElement(
                        "input"
                    );

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


                wrapper.appendChild(
                    label
                );

                wrapper.appendChild(
                    input
                );


                if (parameter.help_text) {
                    const help =
                        document.createElement(
                            "div"
                        );

                    help.className =
                        "probability-field-help";

                    help.textContent =
                        parameter.help_text;

                    wrapper.appendChild(
                        help
                    );
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

                    wrapper.appendChild(
                        error
                    );
                }


                return wrapper;
            }


            function renderParameters(
                spec
            ) {
                parameterFields.innerHTML =
                    "";

                if (!spec.parameters.length) {
                    noParameters.hidden =
                        false;

                    return;
                }

                noParameters.hidden =
                    true;


                spec.parameters.forEach(
                    parameter => {

                        let value =
                            parameter.default;

                        let errorMessage =
                            null;


                        if (
                            firstRender
                            && Object.prototype
                                .hasOwnProperty.call(
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
                            errorMessage =
                                errors.parameters[
                                    parameter.name
                                ];
                        }


                        parameterFields
                            .appendChild(
                                createParameterField(
                                    parameter,
                                    value,
                                    errorMessage
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

                description.textContent =
                    spec.description;

                renderParameters(
                    spec
                );

                firstRender = false;
            }


            distributionSelect
                .addEventListener(
                    "change",
                    () => {
                        firstRender = false;
                        renderDistribution();
                    }
                );


            if (generateAgain) {
                generateAgain
                    .addEventListener(
                        "click",
                        () => {
                            seed.value = "";
                            form.requestSubmit();
                        }
                    );
            }


            if (resetButton) {
                resetButton
                    .addEventListener(
                        "click",
                        () => {
                            firstRender = false;

                            renderParameters(
                                currentDistribution()
                            );

                            if (
                                options.resetFields
                            ) {
                                options.resetFields();
                            }
                        }
                    );
            }


            renderDistribution();
        }


        initializeLab({
            formId:
                "sampling-distribution-form",

            stateId:
                "probability-sampling-state",

            errorsId:
                "probability-sampling-errors",

            distributionId:
                "sampling_distribution",

            parameterFieldsId:
                "sampling-parameter-fields",

            noParametersId:
                "sampling-no-parameters",

            parameterizationId:
                "sampling-parameterization",

            descriptionId:
                "sampling-distribution-description",

            seedId:
                "sampling_seed",

            resetButtonId:
                "sampling-reset-button",

            generateAgainId:
                "sampling-generate-again",

            parameterPrefix:
                "sampling_param_",

            resetFields: () => {
                document.getElementById(
                    "sampling_statistic"
                ).value = "mean";

                document.getElementById(
                    "sampling_sample_size"
                ).value = "30";

                document.getElementById(
                    "sampling_repetitions"
                ).value = "5000";

                document.getElementById(
                    "sampling_seed"
                ).value = "";
            }
        });


        initializeLab({
            formId:
                "sampling-clt-form",

            stateId:
                "probability-clt-state",

            errorsId:
                "probability-clt-errors",

            distributionId:
                "clt_distribution",

            parameterFieldsId:
                "clt-parameter-fields",

            noParametersId:
                "clt-no-parameters",

            parameterizationId:
                "clt-parameterization",

            descriptionId:
                "clt-distribution-description",

            seedId:
                "clt_seed",

            resetButtonId:
                "clt-reset-button",

            generateAgainId:
                "clt-generate-again",

            parameterPrefix:
                "clt_param_",

            resetFields: () => {
                document.getElementById(
                    "clt_sample_sizes"
                ).value =
                    "1, 5, 30, 100";

                document.getElementById(
                    "clt_repetitions"
                ).value =
                    "3000";

                document.getElementById(
                    "clt_seed"
                ).value = "";
            }
        });


        initializeLab({
            formId:
                "sampling-lln-form",

            stateId:
                "probability-lln-state",

            errorsId:
                "probability-lln-errors",

            distributionId:
                "lln_distribution",

            parameterFieldsId:
                "lln-parameter-fields",

            noParametersId:
                "lln-no-parameters",

            parameterizationId:
                "lln-parameterization",

            descriptionId:
                "lln-distribution-description",

            seedId:
                "lln_seed",

            resetButtonId:
                "lln-reset-button",

            generateAgainId:
                "lln-generate-again",

            parameterPrefix:
                "lln_param_",

            resetFields: () => {
                document.getElementById(
                    "lln_max_sample_size"
                ).value =
                    "5000";

                document.getElementById(
                    "lln_paths"
                ).value =
                    "5";

                document.getElementById(
                    "lln_seed"
                ).value = "";
            }
        });

    }
);