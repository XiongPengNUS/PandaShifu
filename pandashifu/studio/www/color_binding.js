// No-jQuery Shiny InputBinding for <input type="color"> (works in Shinylive/static).
(function () {
  function tryRegister() {
    if (!(window.Shiny && Shiny.InputBinding && Shiny.inputBindings)) return false;

    var B = new Shiny.InputBinding();

    function on(el, type, handler) { el.addEventListener(type, handler); }
    function off(el, type, handler) { el.removeEventListener(type, handler); }

    Object.assign(B, {
      find: function (scope) {
        return scope.querySelectorAll(".py-color-input");
      },
      initialize: function (el) {
        if (!el.value) el.value = "#1E90FF";
      },
      getId: function (el) {
        return el.getAttribute("data-input-id") || el.id;
      },
      getValue: function (el) {
        return el.value || null;
      },
      setValue: function (el, value) {
        if (typeof value === "string" && value.startsWith("#")) {
          el.value = value;
          el.dispatchEvent(new Event("input", { bubbles: true }));
          el.dispatchEvent(new Event("change", { bubbles: true }));
        }
      },
      subscribe: function (el, callback) {
        el._pyc_cb = function () { callback(); };
        on(el, "input", el._pyc_cb);
        on(el, "change", el._pyc_cb);
      },
      unsubscribe: function (el) {
        if (el._pyc_cb) {
          off(el, "input", el._pyc_cb);
          off(el, "change", el._pyc_cb);
          delete el._pyc_cb;
        }
      },
      receiveMessage: function (el, data) {
        if (data && Object.prototype.hasOwnProperty.call(data, "value")) {
          this.setValue(el, data.value);
        }
      }
    });

    Shiny.inputBindings.register(B, "mini.color");
    console.log("[mini-color-app] color binding registered");
    return true;
  }

  // Register now or when Shiny connects. Also safe if loaded before/after UI renders.
  if (!tryRegister()) {
    document.addEventListener("shiny:connected", function () { tryRegister(); }, { once: true });
  }
})();