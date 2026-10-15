//! Normalize optional desktop integrations before GTK/WebKit initialization.
//! Do not suppress log output or remove accessibility modules.

#[cfg(target_os = "linux")]
fn xapp_module_exists() -> bool {
    use std::path::PathBuf;
    let mut directories = vec![
        PathBuf::from("/usr/lib/gtk-3.0"),
        PathBuf::from("/usr/lib64/gtk-3.0"),
        PathBuf::from("/usr/local/lib/gtk-3.0"),
    ];
    if let Ok(entries) = std::fs::read_dir("/usr/lib") {
        directories.extend(entries.flatten().map(|e| e.path().join("gtk-3.0")));
    }
    if let Some(paths) = std::env::var_os("GTK_PATH") {
        directories.extend(std::env::split_paths(&paths));
    }
    directories.iter().any(|directory| {
        directory.join("libxapp-gtk3-module.so").is_file()
            || directory.join("modules/libxapp-gtk3-module.so").is_file()
            || std::fs::read_dir(directory).is_ok_and(|entries| {
                entries.flatten().any(|entry| {
                    entry
                        .path()
                        .join("modules/libxapp-gtk3-module.so")
                        .is_file()
                })
            })
    })
}

fn filter_modules(value: &str, xapp_available: bool) -> String {
    value
        .split(':')
        .filter(|module| {
            xapp_available || !matches!(*module, "xapp-gtk3-module" | "libxapp-gtk3-module.so")
        })
        .collect::<Vec<_>>()
        .join(":")
}

pub fn prepare() {
    #[cfg(target_os = "linux")]
    if !xapp_module_exists() {
        for variable in ["GTK3_MODULES", "GTK_MODULES"] {
            if let Ok(value) = std::env::var(variable) {
                let filtered = filter_modules(&value, false);
                if filtered != value {
                    if filtered.is_empty() {
                        std::env::remove_var(variable);
                    } else {
                        std::env::set_var(variable, filtered);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_only_missing_xapp_and_preserves_accessibility() {
        assert_eq!(
            filter_modules("gail:xapp-gtk3-module:atk-bridge", false),
            "gail:atk-bridge"
        );
        assert_eq!(filter_modules("xapp-gtk3-module", false), "");
        assert_eq!(
            filter_modules("xapp-gtk3-module:atk-bridge", true),
            "xapp-gtk3-module:atk-bridge"
        );
        assert_eq!(
            filter_modules("canberra-gtk-module:custom-module", false),
            "canberra-gtk-module:custom-module"
        );
    }
}
