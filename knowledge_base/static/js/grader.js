


// document._keyPushTimeoutId = 0;
// $(document).on('keyup', '.form-control', function(){               
//     var element = $(this);
//     window.clearTimeout(document._keyPushTimeoutId);
//     document._keyPushTimeoutId = setTimeout(function(){
//         element.change();
//     }, 500); 
// }); 

function showHelp(event) {
    let targetHelpIcon = event.target;
    if (targetHelpIcon.classList.contains("grader__help-icon_open")) {
        document.querySelectorAll(".grader__help-icon_open").forEach(icon => icon.classList.remove("grader__help-icon_open"))
    } else {
        document.querySelectorAll(".grader__help-icon_open").forEach(icon => icon.classList.remove("grader__help-icon_open"))
        targetHelpIcon.classList.add("grader__help-icon_open")
    }
}

function graderChexboxChange(event) {
    event.stopPropagation();
    let target = event.target;
    let currentGrader = target.closest(".grader")
    setGraderStatus(currentGrader)
};

function setGraderStatus(currentGrader) {
    const graderFilterIcon = currentGrader.querySelector(".grader__icon_filter");
    const graderUnsoretdIcon = currentGrader.querySelector(".grader__icon_unsort");
    const graderSortAscIcon = currentGrader.querySelector(".grader__icon_ascending");
    const graderSortDescIcon = currentGrader.querySelector(".grader__icon_descending");
    const graderFilterCheckBox = currentGrader.querySelectorAll(".js-filter-checkbox");
    const graderFilterRange = currentGrader.querySelectorAll(".js-filter[type='date']");
    const graderSorters = currentGrader.querySelectorAll(".js-sort");
    const graderResetSort = currentGrader.querySelector(".js-reset-sort");

    console.log(graderFilterCheckBox)
    console.log("CHECKED FIELDS:", [...graderFilterCheckBox].filter(x => x.checked).map(x => x.value));


    let checkedOneCheckBox = Array.prototype.slice.call(graderFilterCheckBox).some(x => x.checked);
    let filledOneCheckBox = Array.prototype.slice.call(graderFilterRange).some(x => x.value);
    if (checkedOneCheckBox || filledOneCheckBox) {
        graderFilterIcon.classList.remove("visually-hidden");
    } else {
        graderFilterIcon.classList.add("visually-hidden");
    };
    graderSorters.forEach(sorter => {
        if (sorter.checked) {
            graderResetSort.checked = false;
            if (sorter.classList.contains("js-sort-ascending")) {
                graderUnsoretdIcon.classList.add("visually-hidden")
                graderSortDescIcon.classList.add("visually-hidden")
                graderSortAscIcon.classList.remove("visually-hidden")
            } else if (sorter.classList.contains("js-sort-descending")) {
                graderUnsoretdIcon.classList.add("visually-hidden")
                graderSortAscIcon.classList.add("visually-hidden")
                graderSortDescIcon.classList.remove("visually-hidden")
            } 
        };
    });
};

function processGraderResetSort(event) {
    event.stopPropagation(); // Предотвращаем всплытие события
    let target = event.target;
    let currentGrader = target.closest(".grader")
    const graderUnsoretdIcon = currentGrader.querySelector(".grader__icon_unsort");
    const graderSortAscIcon = currentGrader.querySelector(".grader__icon_ascending");
    const graderSortDescIcon = currentGrader.querySelector(".grader__icon_descending");
    
    const graderSorters = currentGrader.querySelectorAll(".js-sort");
    graderSorters.forEach(sorter => sorter.checked = false)
    graderUnsoretdIcon.classList.remove("visually-hidden")
    graderSortAscIcon.classList.add("visually-hidden")
    graderSortDescIcon.classList.add("visually-hidden")
}

// document.addEventListener("click", (ev) => {
//     let clicked = ev.target;
//     if (clicked.classList.contains("grader__top") || clicked.closest(".grader__top")) {
//         let grader = clicked.closest(".grader");
//         if (grader.classList.contains("grader_is-open")) {
//             document.querySelectorAll(".grader_is-open").forEach(icon => icon.classList.remove("grader_is-open"))
//         } else {
//             document.querySelectorAll(".grader_is-open").forEach(icon => icon.classList.remove("grader_is-open"))
//             grader.classList.add("grader_is-open")
//         }
//     };
// })
// document.addEventListener("click", (ev) => {
//     let clicked = ev.target;
//     console.log(clicked)
//     // if (clicked.classList.contains("grader__top") || clicked.closest(".grader__top")) {
//     //     let grader = clicked.closest(".grader");
//     //     if (grader.classList.contains("grader_is-open")) {
//     //         document.querySelectorAll(".grader_is-open").forEach(icon => icon.classList.remove("grader_is-open"))
//     //     } else {
//     //         document.querySelectorAll(".grader_is-open").forEach(icon => icon.classList.remove("grader_is-open"))
//     //         grader.classList.add("grader_is-open")
//     //     }
//     // };
//     // if (! clicked.closest(".grader") && document.querySelector(".grader_is-open")) {
//     //     document.querySelectorAll(".grader").forEach(grader => {
//     //         grader.classList.remove("grader_is-open")
//     //     })
//     // }
// })

// Открытие/закрытие по клику на .grader__top
document.querySelectorAll(".grader__top").forEach(graderTop => {
    graderTop.addEventListener("mousedown", e => e.preventDefault()); // предотвратить фокус на input
    graderTop.addEventListener("click", (ev) => {
        const grader = ev.currentTarget.closest('.grader');
        const isOpen = grader.classList.contains('grader_is-open');

        document.querySelectorAll('.grader_is-open').forEach(g => g.classList.remove('grader_is-open'));

        if (!isOpen) {
            grader.classList.add('grader_is-open');
            document.activeElement.blur();
        }
    });
});

// Закрытие всех .grader при клике вне
document.addEventListener("click", (ev) => {
    const clickedInsideGrader = ev.target.closest('.grader');
    if (!clickedInsideGrader) {
        document.querySelectorAll('.grader_is-open').forEach(g => g.classList.remove('grader_is-open'));
    }
});