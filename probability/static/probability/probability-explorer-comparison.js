document.addEventListener(
    "DOMContentLoaded",
    () => {

        const form =
            document.getElementById(
                "probability-comparison-form"
            );

        if (!form) {
            return;
        }


        const config = JSON.parse(
            document.getElementById(
                "probability-ui-config"
            ).textContent
        );

        const initialState = JSON.parse(
            document.getElementById(
                "probability-comparison-state"
            ).textContent
        );

        const initialErrors = JSON.parse(
            document.getElementById(
                "probability-comparison-errors"
            ).textContent
        );


        const categorySelect =
            document.getElementById(
                "comparison_category"
            );

        const viewSelect =
            document.getElementById(
                "comparison_view"
            );

        const curvesContainer =
            document.getElementById(
                "comparison-curves"
            );

        const countInput =
            document.getElementById(
                "comparison_count"
            );

        const addButton =
            document.getElementById(
                "comparison-add-curve"
            );

        const resetButton =
            document.getElementById(
                "comparison-reset-button"
            );


        const minimumCurves =
            initialState.min_curves;

        const maximumCurves =
            initialState.max_curves;


        let curves = JSON.parse(
            JSON.stringify(
                initialState.curves
            )
        );

        let firstRender = true;


        function distributionsForCategory(
            category
        ) {
            return Object
                .values(
                    config.distributions
                )
                .filter(
                    distribution =>
                        distribution.category
                        === category
                );
        }


        function defaultsForCategory(
            category
        ) {
            return JSON.parse(
                JSON.stringify(
                    initialState.defaults[
                        category
                    ]
                )
            );
        }


        function currentSpec(
            distributionKey
        ) {
            return config.distributions[
                distributionKey
            ];
        }


        function comparisonViews() {
            const category =
                categorySelect.value;

            if (category === "discrete") {
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

            const hazardAvailable =
                curves.every(
                    curve => {
                        const spec =
                            currentSpec(
                                curve.distribution
                            );

                        return (
                            spec
                            && spec.supports_hazard
                        );
                    }
                );

            if (hazardAvailable) {
                views.push({
                    key: "hazard",
                    label:
                        "Hazard function"
                });
            }

            return views;
        }


        function renderViews(
            preferredValue = null
        ) {
            const previous =
                preferredValue
                || viewSelect.value
                || initialState.view;

            const views =
                comparisonViews();

            viewSelect.innerHTML = "";

            const defaultView =
                categorySelect.value
                === "continuous"
                    ? "pdf"
                    : "pmf";

            const selected =
                views.some(
                    view =>
                        view.key
                        === previous
                )
                    ? previous
                    : defaultView;

            views.forEach(
                view => {
                    const option =
                        document.createElement(
                            "option"
                        );

                    option.value =
                        view.key;

                    option.textContent =
                        view.label;

                    option.selected =
                        view.key
                        === selected;

                    viewSelect.appendChild(
                        option
                    );
                }
            );
        }


        function createParameterField(
            curve,
            curveIndex,
            parameter
        ) {
            const wrapper =
                document.createElement(
                    "div"
                );

            wrapper.className =
                "probability-dynamic-field";

            const inputId =
                `compare_${curveIndex}`
                + `_param_${parameter.name}`;

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

            let value =
                parameter.default;

            if (
                curve.parameters
                && Object.prototype
                    .hasOwnProperty
                    .call(
                        curve.parameters,
                        parameter.name
                    )
            ) {
                value =
                    curve.parameters[
                        parameter.name
                    ];
            }

            input.value = value;

            input.addEventListener(
                "input",
                event => {
                    curves[
                        curveIndex
                    ].parameters[
                        parameter.name
                    ] = event.target.value;
                }
            );

            wrapper.appendChild(
                label
            );

            wrapper.appendChild(
                input
            );


            const curveErrors =
                firstRender
                    ? initialErrors[
                        String(
                            curveIndex
                        )
                    ]
                    : null;

            const errorMessage =
                curveErrors
                    ? curveErrors[
                        parameter.name
                    ]
                    : null;

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


        function renderCurves() {
            curvesContainer.innerHTML = "";

            const availableDistributions =
                distributionsForCategory(
                    categorySelect.value
                );

            curves.forEach(
                (
                    curve,
                    curveIndex
                ) => {

                    const card =
                        document.createElement(
                            "div"
                        );

                    card.className =
                        "probability-comparison-curve";


                    const header =
                        document.createElement(
                            "div"
                        );

                    header.className =
                        "probability-comparison-curve-header";


                    const title =
                        document.createElement(
                            "strong"
                        );

                    title.textContent =
                        `Curve ${curveIndex + 1}`;


                    const removeButton =
                        document.createElement(
                            "button"
                        );

                    removeButton.type =
                        "button";

                    removeButton.className =
                        (
                            "probability-curve-remove"
                        );

                    removeButton.textContent =
                        "Remove";

                    removeButton.disabled =
                        curves.length
                        <= minimumCurves;

                    removeButton.addEventListener(
                        "click",
                        () => {
                            if (
                                curves.length
                                <= minimumCurves
                            ) {
                                return;
                            }

                            curves.splice(
                                curveIndex,
                                1
                            );

                            firstRender = false;

                            renderCurves();
                            renderViews();
                        }
                    );


                    header.appendChild(
                        title
                    );

                    header.appendChild(
                        removeButton
                    );

                    card.appendChild(
                        header
                    );


                    const topGrid =
                        document.createElement(
                            "div"
                        );

                    topGrid.className =
                        "probability-fields-grid";


                    const distributionField =
                        document.createElement(
                            "div"
                        );

                    distributionField.className =
                        "probability-dynamic-field";


                    const distributionLabel =
                        document.createElement(
                            "label"
                        );

                    distributionLabel.textContent =
                        "Distribution";


                    const distributionSelect =
                        document.createElement(
                            "select"
                        );

                    distributionSelect.name =
                        (
                            `compare_${curveIndex}`
                            + "_distribution"
                        );

                    distributionSelect.className =
                        "probability-control";


                    availableDistributions.forEach(
                        distribution => {

                            const option =
                                document.createElement(
                                    "option"
                                );

                            option.value =
                                distribution.key;

                            option.textContent =
                                distribution.label;

                            option.selected =
                                distribution.key
                                === curve.distribution;

                            distributionSelect
                                .appendChild(
                                    option
                                );
                        }
                    );


                    distributionSelect
                        .addEventListener(
                            "change",
                            event => {

                                const newKey =
                                    event.target.value;

                                const newSpec =
                                    currentSpec(
                                        newKey
                                    );

                                curves[
                                    curveIndex
                                ] = {
                                    distribution:
                                        newKey,

                                    label:
                                        newSpec.label,

                                    parameters:
                                        Object.fromEntries(
                                            newSpec.parameters
                                                .map(
                                                    parameter => [
                                                        parameter.name,
                                                        parameter.default
                                                    ]
                                                )
                                        )
                                };

                                firstRender = false;

                                renderCurves();
                                renderViews();
                            }
                        );


                    distributionField
                        .appendChild(
                            distributionLabel
                        );

                    distributionField
                        .appendChild(
                            distributionSelect
                        );


                    const labelField =
                        document.createElement(
                            "div"
                        );

                    labelField.className =
                        "probability-dynamic-field";


                    const curveLabel =
                        document.createElement(
                            "label"
                        );

                    curveLabel.textContent =
                        "Curve label";


                    const labelInput =
                        document.createElement(
                            "input"
                        );

                    labelInput.type = "text";

                    labelInput.name =
                        (
                            `compare_${curveIndex}`
                            + "_label"
                        );

                    labelInput.className =
                        "probability-control";

                    labelInput.value =
                        curve.label || "";

                    labelInput.addEventListener(
                        "input",
                        event => {
                            curves[
                                curveIndex
                            ].label =
                                event.target.value;
                        }
                    );


                    labelField.appendChild(
                        curveLabel
                    );

                    labelField.appendChild(
                        labelInput
                    );


                    topGrid.appendChild(
                        distributionField
                    );

                    topGrid.appendChild(
                        labelField
                    );

                    card.appendChild(
                        topGrid
                    );


                    const spec =
                        currentSpec(
                            curve.distribution
                        );


                    if (
                        spec
                        && spec.parameters.length
                    ) {
                        const parameterGrid =
                            document.createElement(
                                "div"
                            );

                        parameterGrid.className =
                            (
                                "probability-fields-grid "
                                + "probability-comparison-"
                                + "parameters"
                            );

                        spec.parameters.forEach(
                            parameter => {
                                parameterGrid
                                    .appendChild(
                                        createParameterField(
                                            curve,
                                            curveIndex,
                                            parameter
                                        )
                                    );
                            }
                        );

                        card.appendChild(
                            parameterGrid
                        );
                    }

                    curvesContainer.appendChild(
                        card
                    );
                }
            );


            countInput.value =
                curves.length;

            addButton.disabled =
                curves.length
                >= maximumCurves;
        }


        addButton.addEventListener(
            "click",
            () => {
                if (
                    curves.length
                    >= maximumCurves
                ) {
                    return;
                }

                const available =
                    distributionsForCategory(
                        categorySelect.value
                    );

                const spec =
                    available[0];

                curves.push({
                    distribution:
                        spec.key,

                    label:
                        spec.label,

                    parameters:
                        Object.fromEntries(
                            spec.parameters.map(
                                parameter => [
                                    parameter.name,
                                    parameter.default
                                ]
                            )
                        )
                });

                firstRender = false;

                renderCurves();
                renderViews();
            }
        );


        categorySelect.addEventListener(
            "change",
            () => {
                curves =
                    defaultsForCategory(
                        categorySelect.value
                    );

                firstRender = false;

                renderCurves();
                renderViews();
            }
        );


        resetButton.addEventListener(
            "click",
            () => {
                curves =
                    defaultsForCategory(
                        categorySelect.value
                    );

                firstRender = false;

                renderCurves();
                renderViews();
            }
        );


        renderCurves();

        renderViews(
            initialState.view
        );

        firstRender = false;
    }
);