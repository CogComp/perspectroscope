const COLOR_PROFILE = {
    "sup": {
        "background": "#f5f5f5",
        "persp_color_low": "#d9eaff",
        "persp_color_intense": "#3498db",
    },
    "und": {
        "background": "#f5f5f5",
        "persp_color_low": "#ffdcdf",
        "persp_color_intense": "#e74c3c",
    }
};

const NEW_PERSP_TEMPLATE =  "<div class=\"persp-title btn-lg\">" +
                            "   <div class=\"persp-text\">" +
                            "       <form action='Javascript:void(0)'>" +
                            "           <input class='form-control' placeholder='Write your own perspective here...'>" +
                            "       </form>" +
                            "   </div>" +
                            "</div>";

/**
 * Function to add new perpsectives to container
 * @param persp_ctnr jquery wrapped object for the perspective container
 */
function add_new_perspective(persp_ctnr) {

    let b_locked = $(persp_ctnr).prop('data-new-persp-lock');

    if (b_locked !== 'locked') {
        let _is_ctnr_for = $(persp_ctnr).hasClass("persps-container-for");
        let _color_profile = _is_ctnr_for ? COLOR_PROFILE["sup"] : COLOR_PROFILE["und"];
        let _title = $(persp_ctnr).find("h6:first");
        let new_persp = $(NEW_PERSP_TEMPLATE).clone();
        $(new_persp).css('background-color', _color_profile["persp_color_low"]);
        _title.after(new_persp);

        $(new_persp).find('form').submit(function() {
            let _input_val = $(this).find("input:first").val();
            submit_new_perspective(new_persp, _input_val);
        });

        $(persp_ctnr).prop('data-new-persp-lock', 'locked');
    }
}


/**
 *
 * @param el
 * @returns {jQuery|HTMLElement}
 */
function find_closest_persp_ctnr(el) {
    return $(el).closest(".persps-container");
}

function get_current_claim() {
    return "{{ claim_text }}";
}


function submit_new_perspective(persp_title_el, new_persp) {

    // Step 1: replace the form with input perspective text
    let el_persp_text = $(persp_title_el).find(".persp-text");
    $(el_persp_text).html("<div>" + new_persp + "</div>");

    let _ctnr = find_closest_persp_ctnr(persp_title_el);
    $(_ctnr).prop('data-new-persp-lock', 'unlocked');

    let stance = $(_ctnr).hasClass() ? 'SUP' : 'UND';
    // Step 2: Save the new perspective to db
    {% if tutorial != "true"%}
    let payload = {
        "claim": get_current_claim(),
        "perspective": new_persp,
        "stance": stance,
        "comment": "",
    };

    $.post('/api/submit_new_perspective/', payload, function() {
        console.log(payload);
    });

    {% endif %}
    return false;
}