(function () {
    "use strict";

    function updateDataSummary() {
        const textarea = document.getElementById("data");
        const summary = document.getElementById("control-data-summary");
        if (!textarea || !summary) return;

        const lines = textarea.value
            .replace(/\r/g, "")
            .split("\n")
            .map((line) => line.trim())
            .filter(Boolean);

        if (!lines.length) {
            summary.textContent = "";
            return;
        }

        const counts = lines.map((line) => {
            if (line.includes("\t")) {
                return line.split("\t").filter((value) => value.trim() !== "").length;
            }
            if (line.includes(";")) {
                return line.split(";").filter((value) => value.trim() !== "").length;
            }
            return line.split(/\s+/).filter(Boolean).length;
        });

        const uniqueCounts = [...new Set(counts)];
        const shape = uniqueCounts.length === 1
            ? `${lines.length} row(s) × ${uniqueCounts[0]} value(s)`
            : `${lines.length} row(s) · unequal row lengths`;

        summary.textContent = shape;
    }

    function updateCapabilitySigmaField() {
        const selector = document.getElementById("sigma_method");
        const field = document.getElementById("control-within-sigma-field");
        if (!selector || !field) return;

        field.classList.toggle("is-hidden", selector.value !== "provided");
    }

    function updateVMaskNote() {
        const selector = document.getElementById("cusum_method");
        const note = document.getElementById("control-vmask-note");
        if (!selector || !note) return;

        note.classList.toggle("is-hidden", selector.value !== "vmask");
    }

    document.addEventListener("DOMContentLoaded", function () {
        const textarea = document.getElementById("data");
        if (textarea) {
            textarea.addEventListener("input", updateDataSummary);
            updateDataSummary();
        }

        const sigmaSelector = document.getElementById("sigma_method");
        if (sigmaSelector) {
            sigmaSelector.addEventListener("change", updateCapabilitySigmaField);
            updateCapabilitySigmaField();
        }

        const cusumSelector = document.getElementById("cusum_method");
        if (cusumSelector) {
            cusumSelector.addEventListener("change", updateVMaskNote);
            updateVMaskNote();
        }
    });
}());
