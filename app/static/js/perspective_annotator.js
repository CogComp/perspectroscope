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

const NEW_PERSP_TEMPLATE =  "<div class=\"persp-title btn-lg\"" +
                            "   <div class=\"col persp-text\">" +
                            "       <div>" +
                            "           <form>" +
                            "               <input class='form-control' placeholder='Write your own perspective here...'>" +
                            "           </form>" +
                            "       </div>" +
                            "   </div>" +
                            "</div>";

/**
 * Function to add new perpsectives to container
 * @param persp_ctnr jquery wrapped object for the perspective container
 */
function add_new_perspective(persp_ctnr) {
    let _is_ctnr_for = $(persp_ctnr).hasClass("persps-container-for");
    let _color_profile = _is_ctnr_for ? COLOR_PROFILE["sup"] : COLOR_PROFILE["und"];
    let _title = $(persp_ctnr).find("h6:first");
    let new_persp = $(NEW_PERSP_TEMPLATE).clone();
    $(new_persp).css('background-color', _color_profile["persp_color_low"]);
    _title.after(new_persp);
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

function submit_new_perspective_to_db() {

}