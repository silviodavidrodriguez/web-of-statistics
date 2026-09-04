document.addEventListener(
    "DOMContentLoaded",
    () => {

        const form =
            document.getElementById(
                "probability-simulation-form"
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
                "probability-simulation-state"
            ).textContent
        );

        const errors = JSON.parse(
            document.getElementById(
                "probability-simulation-errors"
            ).textContent
        );


        const distributionSelect =
            document.getElementById(
                "simulation_distribution"
            );

        const parameterFields =
            document.getElementById(
                "simulation-parameter-fields"
            );

        const noParameters =
            document.getElementById(
                "simulation-no-parameters"
            );

        const parameterization =
            document.getElementById(
                "simulation-parameterization"
            );

        const description =
            document.getElementById(
                "simulation-distribution-description"
            );

        const sampleSize =
            document.getElementById(
                "simulation_sample_size"
            );

        const seed =
            document.getElementById(
                "simulation_seed"
            );

        const resetButton =
            document.getElementById(
                "simulation-reset-button"
            );

        const generateAgain =
            document.getElementById(
                "simulation-generate-again"
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
                `simulation_param_${parameter.name}`;


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
            parameterFields.innerHTML = "";

            if (!spec.parameters.length) {
                noParameters.hidden = false;
                return;
            }

            noParameters.hidden = true;


            spec.parameters.forEach(
                parameter => {

                    let value =
                        parameter.default;

                    let errorMessage = null;


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


                    parameterFields.appendChild(
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

                renderParameters(
                    currentDistribution()
                );

                sampleSize.value = "1000";
                seed.value = "";
            }
        );


        generateAgain.addEventListener(
            "click",
            () => {
                seed.value = "";
                form.requestSubmit();
            }
        );


        renderDistribution();


        // ====================================================
        // Copy generated sample
        // ====================================================

        const exportForm =
            document.getElementById(
                "simulation-export-form"
            );

        const copyButton =
            document.getElementById(
                "simulation-copy-data"
            );

        const copyStatus =
            document.getElementById(
                "simulation-copy-status"
            );


        async function writeClipboard(
            text
        ) {
            if (
                navigator.clipboard
                && navigator.clipboard.writeText
            ) {
                await navigator.clipboard.writeText(
                    text
                );

                return;
            }

            const textarea =
                document.createElement(
                    "textarea"
                );

            textarea.value = text;
            textarea.style.position = "fixed";
            textarea.style.opacity = "0";

            document.body.appendChild(
                textarea
            );

            textarea.select();

            document.execCommand(
                "copy"
            );

            textarea.remove();
        }


        if (
            exportForm
            && copyButton
        ) {
            copyButton.addEventListener(
                "click",
                async () => {

                    copyButton.disabled = true;

                    if (copyStatus) {
                        copyStatus.textContent =
                            "Preparing data...";
                    }

                    try {
                        const data =
                            new FormData(
                                exportForm
                            );

                        data.set(
                            "action",
                            "copy"
                        );

                        const response =
                            await fetch(
                                exportForm.action,
                                {
                                    method: "POST",
                                    body: data,
                                    credentials:
                                        "same-origin"
                                }
                            );


                        if (!response.ok) {
                            throw new Error(
                                await response.text()
                            );
                        }


                        const content =
                            await response.text();


                        await writeClipboard(
                            content
                        );


                        if (copyStatus) {
                            copyStatus.textContent =
                                "Copied";
                        }

                    } catch (error) {

                        console.error(
                            error
                        );

                        if (copyStatus) {
                            copyStatus.textContent =
                                "Copy failed";
                        }

                    } finally {

                        copyButton.disabled =
                            false;

                    }
                }
            );
        }
    }
);