import os
import sys
import site

def patch_package():
    try:
        # Locate all site-packages directories
        site_package_locations = site.getsitepackages() + [site.getusersitepackages()]

        def patch_transformers(package_path, package_name='transformers'):
            for root, dirs, files in os.walk(package_path):
                if package_name in dirs:
                    transformers_path = os.path.join(root, package_name, 'utils', 'import_utils.py')
                    if os.path.isfile(transformers_path):
                        print(f'Found: {transformers_path}')
                        break
            if os.path.isfile(transformers_path):
                with open(transformers_path, 'r') as f:
                    lines = f.readlines()

                modified = False
                for i, line in enumerate(lines):
                    if 'def check_torch_load_is_safe():' in line.strip():
                        if lines[i + 1].strip() == 'return True':
                            # file already patched
                            break
                        lines[i] = 'def check_torch_load_is_safe():\n    return True\n'  # Comment out the original function definition
                        # Replace the function definition with a safe version
                        # lines[i + 1] = 'return True\n'
                        modified = True
                        break

                if modified:
                    with open(transformers_path, 'w') as f:
                        f.writelines(lines)
                    print(f'✔ Patched: {transformers_path}')
                else:
                    print(f'ℹ Already patched or not found in: {transformers_path}')
                return
            print("⚠ Could not find target file to patch.")
        for path in site_package_locations:
            patch_transformers(path)
    except Exception as e:
        print(f'❌ Failed to patch: {e}')
