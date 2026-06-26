export const uid = () => crypto.randomUUID();

export function tail(value, maxLength) {
  if (!value || value.length <= maxLength) return value;
  return `...${value.slice(value.length - maxLength + 3)}`;
}
